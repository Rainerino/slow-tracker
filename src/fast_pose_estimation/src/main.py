#! /usr/bin/env python3
import rospy as ros
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import mediapipe as mp
import cv2
import tf2_ros
import numpy as np
from tf import transformations

CAMERA_FRAME_ID = "camera"
WORLD_FRAME_ID = "world"
BGR = "bgr8"

class FastPoseEstimator:
    def __init__(self, pose_rate: float=5.0, goal_rate: float=1.0,  visibility_threshold: float = 0.1, debug_mode=False):
        self.pose_pub_timer = ros.get_time()
        self.depth_update_timer = ros.get_time()
        self.goal_pub_timer = ros.get_time()
        
        # Subscriber
        ros.Subscriber("camera/color/image_raw", Image, self.mediapipe_pose_callback)
        ros.Subscriber("camera/aligned_depth_to_color/image_raw", Image, self.depth_estimator_callback)
        ros.Subscriber(f"{ros.get_name()}/head", PoseStamped, self.set_goal_callback)
        ros.Subscriber("vins_fusion/odometry", Odometry, self.update_odom)

        # Publisher
        self.head_pose_pub = ros.Publisher(f"{ros.get_name()}/head", PoseStamped, queue_size=10)
        self.img_pub = ros.Publisher(f"{ros.get_name()}/image_pose", Image, queue_size=1)
        self.goal_pub = ros.Publisher("move_base_simple/goal", PoseStamped, queue_size=1)

        self.cv_bridge = CvBridge()

        # Config
        self.pose_rate=pose_rate
        self.visibility_threshold=visibility_threshold
        self.debug_mode = debug_mode
        self.goal_rate = goal_rate

        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, enable_segmentation=True, min_tracking_confidence=0.5)

        # Intermediate variables, care for race conditions. 
        self.pose_mask = np.array([])
        self.masked_depth_map = np.array([])
        self.odom = PoseStamped()
        self.next_goal_odom = PoseStamped()

        # TF related.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_1listener = tf2_ros.TransformListener(self.tf_buffer)

    def update_odom(self, odom_msg: Odometry) -> None:
        if ros.get_time() -  self.goal_pub_timer > 1/ self.goal_rate:
            self.goal_pub_timer = ros.get_time()
            self.odom.pose = odom_msg.pose.pose
            self.goal_pub.publish(self.next_goal_odom)

    def set_goal_callback(self, target_pose_msg: PoseStamped) -> None:
        goal_pose_msg = PoseStamped()
        goal_pose_msg.header.frame_id = WORLD_FRAME_ID
        target_pose = np.array([target_pose_msg.pose.position.x, target_pose_msg.pose.position.y, target_pose_msg.pose.position.z])

        # target_ori = np.array([pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z])
        cur_pose = np.array([self.odom.pose.position.x, self.odom.pose.position.y, self.odom.pose.position.z])
        cur_heading = np.array([self.odom.pose.orientation.w, self.odom.pose.orientation.x, self.odom.pose.orientation.y, self.odom.pose.orientation.z])   
        
        # Get vector to target in 2D
        target_pose[2] = 0
        cur_pose[2] = 0
        target_vec = target_pose - cur_pose
        target_yaw_rad = np.arctan2(target_vec[0], target_vec[1])
        
        ros.loginfo(f"{target_yaw_rad}")
        to_target_ori = transformations.quaternion_from_euler(0, 0, -target_yaw_rad+np.pi/2)

        goal_pose = target_pose - target_vec * 2
        
        goal_pose_msg.pose.orientation.x = to_target_ori[0]
        goal_pose_msg.pose.orientation.y = to_target_ori[1]
        goal_pose_msg.pose.orientation.z = to_target_ori[2]
        goal_pose_msg.pose.orientation.w = to_target_ori[3]

        # fix the height at 1.5 meters
        goal_pose_msg.pose.position.x = goal_pose[0]
        goal_pose_msg.pose.position.y = goal_pose[1]
        goal_pose_msg.pose.position.z = 1.5
        # Make the goal to be along the axis
        self.next_goal_odom = goal_pose_msg
        

    def depth_estimator_callback(self, image: Image) -> None:
        # get depth map and estimate pose
        if ros.get_time() - self.depth_update_timer > (1/self.pose_rate)/3:
            self.depth_update_timer = ros.get_time()
            # TODO: Use img pointer instead
            cv_img: cv2.Mat = self.cv_bridge.imgmsg_to_cv2(image)
            if self.pose_mask.shape[0:2] == cv_img.shape[0:2]:
                # TODO: there is a bug here somewhere. Image is not masked correctly. Somewhat normalized?
                self.masked_depth_map = cv2.bitwise_and(cv_img, cv_img, mask=self.pose_mask)

    def mediapipe_pose_callback(self, image: Image) -> None:
        if ros.get_time() - self.pose_pub_timer > (1/self.pose_rate):
            self.pose_pub_timer = ros.get_time()
            cv_img: cv2.Mat = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            results = self.mp_pose.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            # No landmark is found, no need to process anything.
            if not results.pose_world_landmarks:
                return
            condition =  np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
            bg_img = np.zeros(cv_img.shape, dtype=np.uint8)
            masked_color_img = np.where(condition, cv_img, bg_img)
            
            annotated_img = masked_color_img.copy()
            _, self.pose_mask = cv2.threshold(cv2.cvtColor(masked_color_img, cv2.COLOR_BGR2GRAY),1, 255, cv2.THRESH_BINARY)

            # ros.loginfo(f"{self.pose_mask.shape}, {cv_img.shape}")

            mp.solutions.drawing_utils.draw_landmarks(
                        annotated_img,
                        results.pose_world_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

            # Only publish image in debug mode.
            if self.debug_mode:
                try:
                    if self.masked_depth_map.shape[0:2] == annotated_img.shape[0:2]:
                        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.masked_depth_map))
                    elif self.pose_mask.shape[0:2] == annotated_img.shape[0:2]:
                        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv2.flip(self.pose_mask, 1)))
                    else:
                        self.img_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv2.flip(annotated_img, 1), "bgr8"))
                except CvBridgeError as e:
                    ros.logerror(f"error {e}")

            # Change from camera to world frame.
            try:
                trans = self.tf_buffer.lookup_transform(WORLD_FRAME_ID, CAMERA_FRAME_ID, ros.Time())
            except tf2_ros.LookupException as e:
                ros.logerr(f"{e}")
                return
            # ros.loginfo(f"{trans}")
            
            # Treat human body as a flat geometry, averaging over the depth map.

            # A person is a 175 X 50 X 50 box
            # The furthest point is 0, the larger the value, the closer the point is. 
            HUMAN_WIDTH = 25
            MAX_DISTANCE = 12.0
            DEPTH_UNITS = 0.001 # number of meters represented by a single depth unit
            # TODO: bug here, could receive empty frame
            processing_image = self.masked_depth_map.copy()
            if not np.count_nonzero(processing_image):
                # No depth image found, skip this frame.
                return
            normalize = np.sum(processing_image) / np.count_nonzero(processing_image)
            filtered = np.where(processing_image < normalize + HUMAN_WIDTH, processing_image, 0)
            remapped_depth = (255 - filtered) / 255 / DEPTH_UNITS # Map from Gray scale uint8 to meters.
            remapped_depth = cv2.GaussianBlur(remapped_depth, (5, 5), cv2.BORDER_DEFAULT)
            depth_estimated = (np.sum(remapped_depth) / np.count_nonzero(remapped_depth) ) / 10000.0 # TODO: magic number LOL

            # Construct transformation matrix
            head_pose_msg: PoseStamped = PoseStamped()
            head_pose_msg.header.frame_id = WORLD_FRAME_ID

            T = np.zeros((4, 4))
            R = transformations.quaternion_matrix([trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z])
            # ros.loginfo(R)
            T[0:3, 0:3] = R[0:3, 0:3]
            T[3, 0:3] = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            T[3, 3] = 1.0

            right_hand_pose = np.array([results.pose_world_landmarks.landmark[0].x, results.pose_world_landmarks.landmark[0].y, results.pose_world_landmarks.landmark[0].z, 1.0])

            head_translated = np.matmul(T, right_hand_pose)

            ros.logdebug(f"transformation matrix: {T}, pose {right_hand_pose}, trans {head_translated}")

            head_pose_world = np.array([depth_estimated, -head_translated[1] * 2, head_translated[0] * 2])
            ros.loginfo(f"{head_pose_world}")

            if results.pose_world_landmarks.landmark[0].visibility > self.visibility_threshold:
                head_pose_msg.pose.position.x = head_pose_world[0]
                head_pose_msg.pose.position.y = head_pose_world[1]
                head_pose_msg.pose.position.z = head_pose_world[2]
                self.head_pose_pub.publish(head_pose_msg)

    
if __name__=='__main__':
    ros.init_node('fast_pose_est')
    fast_estimator: FastPoseEstimator = FastPoseEstimator()
    ros.spin()