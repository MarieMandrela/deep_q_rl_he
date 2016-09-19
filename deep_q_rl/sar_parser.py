import numpy as np
import matplotlib.image as mpimg
import cv2
import ale_data_set
import glob
import os

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8

class SARParser:

    def __init__(self, data_set, min_action_set, resize_method, width, height, gray_scale):
        self.gray_scale = gray_scale

        self.width = width
        self.height = height

        self.resized_width = data_set.width
        self.resized_height = data_set.height
        self.resize_method = resize_method

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)

        self.data_set = data_set

        self.min_action_set = min_action_set

    @staticmethod
    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def import_episodes(self, screenshots_dir, rewards_dir):
        directories = self.get_immediate_subdirectories(screenshots_dir)
        for i in directories:
            rewards_file = rewards_dir + i + ".csv"
            screenshot_dir = screenshots_dir + i + "/"
            self.import_episode(screenshot_dir, rewards_file)
        return self.data_set

    def import_episode(self, screenshot_dir, rewards_file):

        screenshot_names = glob.glob1(screenshot_dir, "*.png")
        screenshot_num = len(screenshot_names)

        assert (screenshot_num > self.buffer_length), \
            "Need to import more SARs than buffer length(" + str(self.buffer_length) + ")"

        assert (self.data_set.size + screenshot_num < self.data_set.max_steps),\
            "Can't save SARs in the given data_set as it doesn't have enough space left."

        # get actions and rewards as numpy array form csv file
        action_rewards = np.genfromtxt(rewards_file, delimiter=",")

        # get all image files as numpy array
        self.import_screen(screenshot_dir + '/' + screenshot_names[0])
        for i in xrange(1, screenshot_num - 1):
            self.import_screen(screenshot_dir + '/' + screenshot_names[i])
            observation = self.get_observation()
            min_action = np.where(self.min_action_set == action_rewards[i][0])[0][0]
            action = min_action
            reward = action_rewards[i+1][1]
            self.data_set.add_sample(observation, action, reward, True if i == screenshot_num - 2 else False)

        return self.data_set


    def import_screen(self, file_name):

        img = mpimg.imread(file_name)
        img *= 255
        img = img.astype(np.uint8)

        index = self.buffer_count % self.buffer_length

        self.screen_buffer[index, ...] = img

        self.buffer_count += 1

    # THESE FUNCTIONS ARE JUST COPIED FROM ale_experiment

    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
            crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'crop':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')
