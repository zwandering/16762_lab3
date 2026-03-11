import cv2
import yaml
import rclpy
import os.path as osp
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import detection_utils
import message_filters
import numpy as np


# Don't forget to start the camera before starting this node!
# Part 1: using in-gripper camera
#    ros2 launch stretch_core d405_basic.launch.py
# Part 2: using head camera
#    ros2 launch stretch_core d435i_low_resolution.launch.py
#
# ros2 run rviz2 rviz2 -d `ros2 pkg prefix --share stretch_calibration`/rviz/stretch_simple_test.rviz


class YOLOEObjectDetector(Node):
    def __init__(self, obj_queries):
        super().__init__('yoloe_object_detector')
        self.visualize = True

        # ----------- Camera Streaming Setup -----------

        # subscribe to the robot's color and aligned depth camera image topics from the gripper camera
        # using message_filters, instead of self.create_subscription() to allow us
        # to synchronize the two camera streams can use a single callback that triggers when both come in
        # TODO: ------------- start --------------
        # leave as is for part 1, 
        # change for part 2 to use the head camera
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.color_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')

        # ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true  pointcloud.enable:=false
        # self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        # self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        # self.color_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')

        # TODO: -------------- end ---------------
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_cam_info = None

        # Use ApproximateTimeSynchronizer and register a callback function that runs within some time tolerance of when both images are received
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.color_cam_info_sub],
            queue_size=10,
            slop=0.01  # 10ms tolerance
        )
        self.synchronizer.registerCallback(self.image_callback)

        # bridge to convert ROS2 image messages to OpenCV images
        self.bridge = CvBridge()

        # -----------------------------------------------------

        # ----------- YOLO-E Object Detection SetuP -----------

        # Load the YOLOE model, which should already saved to common models directory on the robot
        #   we use yolo-e-v26-small for its high performance and low latency on limited compute
        model_path = '/home/hello-robot/models'
        model_name = 'yoloe-26s-seg.pt'
        self.model = YOLO(osp.join(model_path, model_name))
        # model_name = './yoloe-26s-seg.pt'
        # self.model = YOLO(model_name)

        # pass prompt for the object/s you want to detect
        self.obj_queries = obj_queries
        self.model.set_classes(self.obj_queries)

        # Run the detector and goals at a fixed frequency to reduce latency introduced by the detector
        #   and give the robot time to move between poses
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_goals_callback)
        self.goal_pub = self.create_publisher(PoseStamped, '/object_detector/goal_pose', 10)
        self.pcd_pub = self.create_publisher(PointCloud2, '/object_detector/target_pointcloud', 10)
        self.goal_pose_msg = None

        # -----------------------------------------------------

    def image_callback(self, color_msg, depth_msg, color_cam_info_msg):
        # convert the color and depth ROS2 image messages to OpenCV images
        # TODO: ------------- start --------------
        # in part 1,fill with your response
        #   you may need to nest things in a try, except in case frames are missing
        #.  if you are unpacking frames correctly, you should see the live color and depth output
        #   plotted in a cv2 window by detection_utils.visualize_detection_masks()
        # in part 2, you may need to make changes to the code to handle the head camera orientation

        try:
            # Convert ROS2 Image messages to OpenCV format
            self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # translate from bgr to rgb for YOLO-E
            self.latest_color = cv2.cvtColor(self.latest_color, cv2.COLOR_BGR2RGB)
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.latest_color_cam_info = color_cam_info_msg
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # TODO: -------------- end ---------------


    def publish_goals_callback(self):
        # run object detection on the RGB image
        # TODO: ------------- start --------------
        # fill with your response
        #   pass the color frame to YOLO-E, parse the results using detection_utils.parse_results()

        if self.latest_color is None:
            detections = None
        else:
            # Run YOLO-E object detection on the color frame
            results = self.model(self.latest_color, verbose=False)
            # Parse the detection results
            detections = detection_utils.parse_results(results)

        # TODO: -------------- end ---------------

        # create visualizations from the detections
        if self.visualize:
            detection_utils.visualize_detections_masks(
                # TODO: minor - change the part= arg when you edit your code for part 2! 
                #   adjusts the color scaling of the depth image display to match the camera range
                part=1, detections=detections, rgb_image=self.latest_color, depth_image=self.latest_depth)

        # get the goal pose and publish it, if it exists
        self.get_goal_pose(detections)

        if self.goal_pose_msg is None:
            print("OBJECT NOT DETECTED, no pose to publish")
            return
        else:
            self.goal_pub.publish(self.goal_pose_msg)
            print()
            print("---------- Published Goal Pose ----------")




    def get_goal_pose(self, detections, target_idx=0):
        if detections is None or len(detections) == 0:
            self.goal_pose_msg = None
            return None

        # TODO: ------------- start --------------
        # in part 1, fill with your response
        #   find the depth at the centroid and project it to 3D using detection_utils.pixel_to_3d()
        #   convert that pose to a PoseStamped msg using detection_utils.get_pose_msg()
        #   save that message to self.goal_pose_msg
        # in part 2, edit the code you wrote for part 1 to now project all points in the mask to 3D,
        #   then get the centroid of the resulting pointcloud to use as the goal pose (instead of the 2D centroid in part 1)

        # Get the target detection
        detection = detections[target_idx]
        poly = detection["mask"]  # (N, 2) polygon points in [x, y] pixel format

        # Rasterize the polygon to get all pixel coordinates inside the mask
        h, w = self.latest_depth.shape
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 1)

        # Get pixel coordinates of all points inside the mask with valid depth
        ys, xs = np.where((poly_mask == 1) & (self.latest_depth > 0))

        if len(xs) == 0:
            self.goal_pose_msg = None
            return None

        # Project each mask pixel to 3D and collect the pointcloud
        camera_mat = np.reshape(self.latest_color_cam_info.k, (3, 3))
        f_x, c_x = camera_mat[0, 0], camera_mat[0, 2]
        f_y, c_y = camera_mat[1, 1], camera_mat[1, 2]
        z_vals = self.latest_depth[ys, xs] / 1000.0  # mm -> m
        x_vals = (xs - c_x) * z_vals / f_x
        y_vals = (ys - c_y) * z_vals / f_y

        # Compute the 3D centroid of the pointcloud
        xyz_3d = np.array([x_vals.mean(), y_vals.mean(), z_vals.mean()])
        print(xyz_3d * 100, 'cm')

        # Publish the target object pointcloud
        points = np.column_stack((x_vals, y_vals, z_vals)).astype(np.float32)
        pcd_msg = PointCloud2()
        pcd_msg.header = Header()
        pcd_msg.header.stamp = self.get_clock().now().to_msg()
        pcd_msg.header.frame_id = 'camera_color_optical_frame'
        pcd_msg.height = 1
        pcd_msg.width = len(points)
        pcd_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        pcd_msg.is_bigendian = False
        pcd_msg.point_step = 12  # 3 x float32
        pcd_msg.row_step = pcd_msg.point_step * pcd_msg.width
        pcd_msg.is_dense = True
        pcd_msg.data = points.tobytes()
        self.pcd_pub.publish(pcd_msg)

        # Convert to PoseStamped message
        self.goal_pose_msg = detection_utils.get_pose_msg(
            timestamp=self.get_clock().now().to_msg(),
            frame_id='camera_color_optical_frame',
            xyz_out=xyz_3d
        )
        
        # TODO: -------------- end ---------------


if __name__ == '__main__':
    rclpy.init()

    # load in the full list of object queries from the yaml file, as well as a target (if specified)
    with open('object_queries.yaml', 'r') as file:
        config = yaml.safe_load(file)
        obj_queries = config['queries']

    yolo_object_detector = YOLOEObjectDetector(obj_queries)
    rclpy.spin(yolo_object_detector)
    yolo_object_detector.destroy_node()
    rclpy.shutdown()
