#! /usr/bin/env python3
import rospy as ros
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import mediapipe as mp
import cv2
import tf2_ros
import numpy as np
from tf import transformations

CAMERA_FRAME_ID = "camera"
WORLD_FRAME_ID = "world"
BGR = "bgr8"

class FastPoseEstimator:
    def __init__(self, pose_rate: float=10.0, visibility_threshold: float = 0.1, debug_mode=False):
        self.pose_pub_timer = ros.get_time()
        self.depth_update_timer = ros.get_time()
        
        # subscriber
        ros.Subscriber("camera/color/image_raw", Image, self.mediapipe_pose_callback)
        ros.Subscriber("camera/aligned_depth_to_color/image_raw", Image, self.depth_estimator_callback)
        

        # publisher
        self.left_hand_pub = ros.Publisher(f"{ros.get_name()}/left_hand", PoseStamped, queue_size=10)
        self.right_hand_pub = ros.Publisher(f"{ros.get_name()}/right_hand", PoseStamped, queue_size=10)
        self.img_pub = ros.Publisher(f"{ros.get_name()}/image_pose", Image, queue_size=1)
        self.cv_bridge = CvBridge()

        self.pose_rate=pose_rate
        self.visibility_threshold=visibility_threshold
        self.debug_mode = debug_mode

        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, enable_segmentation=True, min_tracking_confidence=0.5)

        # Intermediate variables.
        self.pose_mask = np.array([])
        self.masked_depth_map = np.array([])

        # TF related.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_1listener = tf2_ros.TransformListener(self.tf_buffer)

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

            # change from camera to world frame
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
                return
            normalize = np.sum(processing_image) / np.count_nonzero(processing_image)
            filtered = np.where(processing_image < normalize + HUMAN_WIDTH, processing_image, 0)
            remapped_depth = (255 - filtered) / 255 / DEPTH_UNITS
            # remapped_depth = cv2.GaussianBlur(remapped_depth, (5, 5), cv2.BORDER_DEFAULT)
            # filtered_depth_map = cv2.bilateralFilter(self.masked_depth_map, 15, 80, 80)
            # depth_estimated = np.average(remapped_depth)
            depth_estimated = (np.sum(remapped_depth) / np.count_nonzero(remapped_depth) ) / 10000.0 # TODO: magic number LOL
            ros.loginfo(f"before {depth_estimated}")

            # ros.loginfo(f"{remapped_depth}")
            # human_bbox = np.where(self.masked_depth_map < depth_estimated + HUMAN_WIDTH, self.masked_depth_map, 0 )
            
            # depth_estimated = np.sum(human_bbox) / np.count_nonzero(human_bbox)
            # ros.loginfo(f"{depth_estimated}")

            # Construct transformation matrix
            left_hand_pose_msg: PoseStamped = PoseStamped()
            left_hand_pose_msg.header.frame_id = WORLD_FRAME_ID
            right_hand_pose_msg: PoseStamped = PoseStamped()
            right_hand_pose_msg.header.frame_id = WORLD_FRAME_ID

            T = np.zeros((4, 4))
            R = transformations.quaternion_matrix([trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z])
            # ros.loginfo(R)
            T[0:3, 0:3] = R[0:3, 0:3]
            T[3, 0:3] = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            T[3, 3] = 1.0

            right_hand_pose = np.array([results.pose_world_landmarks.landmark[0].x, results.pose_world_landmarks.landmark[0].y, results.pose_world_landmarks.landmark[0].z, 1.0])

            right_hand_translated = np.matmul(T, right_hand_pose)


            # ros.loginfo(f"transformation matrix: {T}, pose {right_hand_pose}, trans {right_hand_translated}")

            if results.pose_world_landmarks.landmark[0].visibility > self.visibility_threshold:
                right_hand_pose_msg.pose.position.x = depth_estimated
                right_hand_pose_msg.pose.position.y = -right_hand_translated[1] * 2
                right_hand_pose_msg.pose.position.z = right_hand_translated[0] * 2
                self.right_hand_pub.publish(right_hand_pose_msg)
            # if results.pose_world_landmarks.landmark[16].visibility > self.visibility_threshold:
            #     left_hand_pose_msg.pose.position.x = results.pose_world_landmarks.landmark[16].y*1
            #     left_hand_pose_msg.pose.position.y = results.pose_world_landmarks.landmark[16].x*1
            #     left_hand_pose_msg.pose.position.z = results.pose_world_landmarks.landmark[16].z*1
            #     self.left_hand_pub.publish(left_hand_pose_msg)

    
if __name__=='__main__':
    ros.init_node('fast_pose_est')
    fast_estimator: FastPoseEstimator = FastPoseEstimator()
    ros.spin()