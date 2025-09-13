import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
from object_mapping import ObjectMapping
from threading import Thread


class Point:
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def __repr__(self):
        return f'({self.x1}, {self.x2}, {self.x3})'

    def to_array(self):
        return np.array([self.x1, self.x2, self.x3])

    def normalize(self):
        self.x1 = self.x1 / np.linalg.norm(self.to_array())
        self.x2 = self.x2 / np.linalg.norm(self.to_array())
        self.x3 = self.x3 / np.linalg.norm(self.to_array())
        return self


class Projection:
    def __init__(self, y1, y2):
        self.y1 = y1
        self.y2 = y2

    def __repr__(self):
        return f'({self.y1}, {self.y2})'

    def to_array(self):
        return np.array([self.y1, self.y2])


class VideoStream:
    # From https://stackoverflow.com/questions/55099413/python-opencv-streaming-from-camera-multithreading-timestamps/55131226
    def __init__(self, src=0, frame_width=800, frame_height=600):
        self.capture = cv2.VideoCapture(src)
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Start capture thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

        # Ensure capture has started
        while self.status is False:
            time.sleep(.1)

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            time.sleep(.05)

        self.capture.release()
        cv2.destroyAllWindows()
        exit(1)

    def release(self):
        self.capture.release()

    def get_frame(self):
        print(f'{self.frame.shape}')
        return [self.frame]


class Camera:
    def __init__(self, position: Point, look_at: Point = None, focal_length=1.0, hardware_id=None, frame_width=800, frame_height=600):
        """
        Initializes a new camera object with the given position and look-at direction.
        :param position: The position of the camera in the global coordinate system.
        :param look_at: The direction that the camera is looking at in the global coordinate system.
        :param focal_length: The focal length of the camera lens, which determines the field of view.
        """
        self.position = position
        self.look_at = look_at.normalize()
        self.focal_length = focal_length
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Compute homogeneous transformation matrix
        self.T = np.zeros((4,4))
        self._homogeneous_transform()

        # Instantiate video stream component from CV2
        self.video_stream = None
        if hardware_id is not None:
            self.video_stream = VideoStream(hardware_id, frame_width=self.frame_width, frame_height=self.frame_height)

    def _homogeneous_transform(self):
        """
        Compute the homogeneous transformation matrix for this camera.
        The camera is defined by its position and a direction vector it's looking at.
        The rotation matrix is computed from these two vectors. The translation matrix is
        then computed to center the camera at the given position.
        """

        # Camera Z
        zc = self.look_at.to_array()

        # Camera X
        z_global = np.array([0, 0, 1])
        xc = np.cross(zc, z_global)
        xc = xc / np.linalg.norm(xc)

        # Camera Y
        yc = np.cross(xc, zc)

        # Construct rotation matrix
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[0, 0:3] = xc
        rotation_matrix[1, 0:3] = yc
        rotation_matrix[2, 0:3] = zc

        # Construct homogeneous transform matrix
        self.T[0:3, 0:3] = rotation_matrix
        self.T[0:3, 3] = -1 * np.matmul(rotation_matrix, self.position.to_array())
        self.T[3, 3] = 1.

    def global_to_camera(self, point: Point) -> Point:
        """
        Convert a point from the global coordinate system to the camera's local coordinate system.

        :param point: The point in the global coordinate system.
        :return: The point in the camera's local coordinate system.
        """

        point_ext = np.concatenate([point.to_array().transpose(), np.array([1])], 0)
        p_camera = np.matmul(self.T, point_ext)
        return Point(p_camera[0], p_camera[1], p_camera[2])

    def project_point(self, point: Point) -> Projection:
        """
        Compute Pinhole projection of Point.

        :param point: The point in the camera's local coordinate system.
        :return: The projection of the point in the camera's image.
        """
        return Projection((self.focal_length/point.x3) * point.x1, (self.focal_length/point.x3) * point.x2)

    def image_from_global(self, point: Point) -> Projection:
        """
        Compute Pinhole projection of Point in the global coordinate system.

        :param point: The point in the global coordinate system.
        :return: The projection of the point in the camera's image.
        """

        return self.project_point(self.global_to_camera(point))


class InverseProblem:
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras

    def compute_inverse(self, projection: list[Projection]):
        M = None

        # Create linear system for the inverse problem
        for idx, cam in enumerate(self.cameras):
            m_cam = np.kron(projection[idx].to_array().reshape((2,1)), cam.T[2, 0:4]) - cam.focal_length * cam.T[0:2, 0:4]
            if M is None:
                M = m_cam
            else:
                M = np.concatenate([M, m_cam], 0)

        # Solve linear system for the inverse problem
        pg = np.linalg.lstsq(M[:, 0:-1], -1*M[:, -1], 1)[0]

        return Point(pg[0], pg[1], pg[2])


def exit_handler(cameras):
    print('My application is ending!')
    for camera in cameras:
        camera.video_stream.release()
    cv2.destroyAllWindows()


def plot_reconstruction(ax, reconstruction_points, object_mapping):
    if len(reconstruction_points) > 0:
        x, y, z = [p.x1 for p in reconstruction_points], [p.x2 for p in reconstruction_points], [p.x3 for p in reconstruction_points]

        ax.cla()  # clear previous frame

        colors = object_mapping.colors

        ax.scatter(x, y, z, c=colors, marker="o")
        if object_mapping.junctions:
            for junction in object_mapping.junctions:
                ax.plot([x[junction[0]], x[junction[1]]], [y[junction[0]], y[junction[1]]], [z[junction[0]], z[junction[1]]], 'k')

        # Axis limits to be adjusted according to scene
        ax.set_xlim(-0.25, .25)
        ax.set_ylim(0, 0.5)
        ax.set_zlim(0.25, 0.75)

        plt.draw()
        plt.pause(0.005)  # tiny pause just to refresh


def main():
    focal_length = 1.08  # From BRIO 100 specs

    # Instantiate cameras
    cameras = [
        Camera(position=Point(0, 0, 0.5), look_at=Point(0, 1, 0), focal_length=focal_length, hardware_id=0, frame_width=400, frame_height=300),
        Camera(position=Point(-0.5, 0, 0.5), look_at=Point(1, 1, 0), focal_length=focal_length, hardware_id=1, frame_width=400, frame_height=300)
    ]
    # cameras = [
    #     Camera(position=Point(1.5, .3, 1.7), look_at=Point(0, 1, 0), focal_length=focal_length, hardware_id=0),
    #     Camera(position=Point(.2, 1.5, 1.2), look_at=Point(1, 1, 0), focal_length=focal_length, hardware_id=1)
    # ]

    # Instantiate InverseProblem object
    inverse_problem = InverseProblem(cameras)

    # Instantiate Object Mapping
    object_type = "hand"
    object_mapping = ObjectMapping(object_type)

    # Load YOLO model
    # model = YOLO('yolo-Weights/yolo11s-pose.pt')
    model = None
    if object_type == "body":
        model = YOLO('yolo-Weights/yolo11s-pose.pt')
    elif object_type == "hand":
        model = YOLO('yolo-Weights/hands_detection.pt')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # body, = ax.plot([], [], [], "ro")  # single point
    scatter = ax.scatter([], [], [], c="r", marker="o")

    plt.ion()
    plt.show()

    try:
        while True:
            # Read frames from camera streams
            camera_frames = [[x for x in camera.video_stream.get_frame()] for camera in cameras]
            frames = [camera_read for camera_read in camera_frames]

            # Pose detection
            results = [model.track(frame, classes=0, conf=0.8, imgsz=480) for frame in frames]

            # Display Video Streams
            for cam_idx in range(len(cameras)):
                cv2.imshow(f"Live Camera {cam_idx}", results[cam_idx][0].plot())

            # Collect points in the image
            points = []
            for cam_idx, result_cam in enumerate(results):
                points_cam = []

                for result in [result_cam[0]]:
                    if len(result.keypoints.xy) > 1:
                        print(f'Camera {cam_idx} detected {len(result.keypoints.xy)} objects.')

                    xy = result.keypoints.xy  # x and y coordinates
                    xyn = result.keypoints.xyn  # normalized
                    kpts = result.keypoints.data  # x, y, visibility (if available)

                    if len(xy) > 0:
                        for idx in range(len(xy[0])):
                            x, y = xyn[0][idx][0], xyn[0][idx][1]
                            x, y = (x - 0.5), (0.5 - y)
                            visibility = kpts[0][idx]
                            points_cam.append(Projection(x, y))

                points.append(points_cam)

            # Compute the 3D reconstruction for set of points from all cameras
            n_points = min([len(points_list) for points_list in points])
            reconstruction = [inverse_problem.compute_inverse([points[cam_idx][pt_idx] for cam_idx in range(len(points))]) for pt_idx in range(n_points)]

            # Plot reconstructed object
            plot_reconstruction(ax, reconstruction, object_mapping)

            # Break if key pressed
            if cv2.waitKey(1) == ord('q'):
                break

            time.sleep(0.1)

    except Exception as e:
        if cameras:
            exit_handler(cameras)
        print(e)
    finally:
        exit_handler(cameras)


if __name__ == '__main__':
    main()
