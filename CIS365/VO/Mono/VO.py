import cv2
import numpy as np
import os


class MonoVo:
    def __init__(
            self,
            image_path,
            pose_path,
            focal_length=718.8560,
            pp=(607.1928, 185.2157),
            lk_params=dict(
                win_size=(21, 21),
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT, 30, 0.01)),
            detector=cv2.FastFeatureDetector_create(
                threshold=25, nonmaxSuppression=True)):
        '''
        The init is the basic constructor for the visual odometry system.
        This class has the following characteristics:

        Parameters
        ----------
        image_path {str} - The file path for the video image sequences
        pose_path {str} - The file path for the true poses in the images
        focal_length {float} - The focal length of the camera used
        pp {tuple} - The principal point of the camera
        lk_params {dict} - The params for the Lucas Kanade optical flow
        detector {cv.FeatureDetector} - The cv2 feature detection algorithm
        '''
        self.image_path = image_path
        self.pose_path = pose_path
        self.focal_length = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0

        try:
            if not all(['.png' in x for x in os.listdir(self.image_path)]):
                raise ValueError('Image path has incorrect images in it!')
        except Exception as e:
            print(e)
            raise ValueError('Image path not found')

        try:
            with open(self.pose_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError('The pose file path is invalid')

        self.process_frame()

    def has_next_frame(self):
        return self.id < len(os.listdir(self.file_path))

    def detect(self, img):
        '''
        Detect features and parse them into an array

        Parameters
        ----------
        img {np.ndarray} - The image to detect on

        Returns
        -------
        np.ndarray - The sequence of ay points denoting the keypoints
        '''

        points = self.detector.detect(img)

        return np.array([
            x.pt for x in points], dtype=np.float32).reshape(-1, 1, 2)

    def visual_odometry(self):
        '''
        The VO code. If a feature falls out of frame to the points that there
        are < 2000 remaining, then we detect new features again
        '''
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        # Calculate optical flow between frames which is the pattern of motion
        # between successive frames
        self.p1, status, err = cv2.calcOpticalFlowPyrLK(
                self.old_frame,
                self.current_frame,
                self.points,
                None,
                **self.lk_params)

        # Save only the good points
        self.good_old = self.p0[status == 1]
        self.good_new = self.p1[status == 1]

        # If the current frame is one of our initial two, we need to change our
        # operation since our behavior is different initially.
        if self.id < 2:
            E, _ = cv2.findEssentialMat(
                    self.good_new,
                    self.good_old,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999, 1.0, None)
            _, self.R, self.t, _ = cv2.recoverPose(
                    E,
                    self.good_old,
                    self.good_new,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999, 1.0, None)
        else:
            E, _ = cv2.findEssentialMat(
                    self.good_new,
                    self.good_old,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999, 1.0, None)
            _, R, t, _ = cv2.recoverPose(
                    E,
                    self.good_old,
                    self.good_new,
                    self.focal,
                    self.pp,
                    cv2.RANSAC,
                    0.999, 1.0, None)
            absolute_scale = self.get_absolute_scale()

            if (absolute_scale > 0.1 and
                    abs(t[2][0]) > abs(t[0][0]) and
                    abs(t[2][0]) > abs([1][0])):
                self.t = self.t + absolute_scale * self.R.dot(t)
                self.R = R.dot(self.R)

            self.n_features = self.good_new.shape[0]

    def get_mono_coordinates(self):
        diag = np.array([-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()

    def get_true_coordinates(self):
        return self.true_coord.flatten()

    def get_absolute_scale(self):
        '''
        Get pose estimation scale for multiplying with the
        rotation and translation vectors
        '''
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

        return np.linalg.norm(true_vect - prev_vect)

    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''

        if self.id < 2:
            self.old_frame = cv2.imread(
                    self.image_path + str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(
                    self.image_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(
                    self.image_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1
