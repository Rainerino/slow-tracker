import rospy as ros
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
import mediapipe as mp
import cv2


class FastPoseEstimator:
    def __init__(self, pose_rate: float=0.25, visibility_threshold: float = 0.5):
        self.pub_time = ros.get_time()
        ros.Subscriber("camera/color/image_raw", Image, self.mediapipe_pose_callback)
        self.left_hand_pub = ros.Publisher(f"{ros.get_name()}/left_hand", PoseStamped, queue_size=10)
        self.right_hand_pub = ros.Publisher(f"{ros.get_name()}/right_hand", PoseStamped, queue_size=10)
        self.img_pub = ros.Publisher(f"{ros.get_name()}/image_pose", Image, queue_size=1)
        self.cv_bridge = CvBridge()
        self.pose_rate=pose_rate
        self.visibility_threshold=visibility_threshold
        
        
    def mediapipe_pose_callback(self, image: Image) -> None:
        if ros.get_time() - self.pub_time > self.pose_rate:
            self.pub_time = ros.get_time()
            cv_img: cv2.Mat = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            annotated_img = cv_img.copy()
            mp_pose = mp.solutions.pose.Pose()
            results = mp_pose.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            mp.solutions.drawing_utils.draw_landmarks(
                        annotated_img,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
            try: 
                annotated_msg: Image = self.cv_bridge.cv2_to_imgmsg(cv2.flip(annotated_img, 1), "bgr8")
                # annotated_img.header.frame_id = "camera"
                self.img_pub.publish(annotated_msg)
            except CvBridgeError as e:
                ros.logerror(f"error {e}")
            
            if results.pose_landmarks is not None:
                # left hand
                ros.loginfo(results.pose_landmarks.landmark[15])
                # right hand
                ros.loginfo(results.pose_landmarks.landmark[16])
                ros.logwarn("asd")
            else:
                return
            
            left_hand_pose_msg: PoseStamped = PoseStamped()
            left_hand_pose_msg.header.frame_id = "map"
            right_hand_pose_msg: PoseStamped = PoseStamped()
            right_hand_pose_msg.header.frame_id = "map"
            
            if results.pose_landmarks.landmark[15].visibility > self.visibility_threshold:
                right_hand_pose_msg.pose.position.x = results.pose_landmarks.landmark[15].y*1
                right_hand_pose_msg.pose.position.y = results.pose_landmarks.landmark[15].x*1
                right_hand_pose_msg.pose.position.z = results.pose_landmarks.landmark[15].z*1
                self.right_hand_pub.publish(right_hand_pose_msg)
            if results.pose_landmarks.landmark[16].visibility > self.visibility_threshold:
                left_hand_pose_msg.pose.position.x = results.pose_landmarks.landmark[16].y*1
                left_hand_pose_msg.pose.position.y = results.pose_landmarks.landmark[16].x*1
                left_hand_pose_msg.pose.position.z = results.pose_landmarks.landmark[16].z*1
                self.left_hand_pub.publish(left_hand_pose_msg)
    
    
if __name__=='__main__':
    ros.init_node('fast_pose_est')
    fast_estimator: FastPoseEstimator = FastPoseEstimator()
    ros.spin()