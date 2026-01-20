import cv2
import os
import glob
import numpy as np
import aslam_cv as acv
import sm


class FolderImageDatasetReaderIterator(object):
  """Iterator for folder-based image dataset"""
  def __init__(self, dataset, indices=None):
    self.dataset = dataset
    if indices is None:
      self.indices = np.arange(dataset.numImages())
    else:
      self.indices = indices
    self.iter = self.indices.__iter__()

  def __iter__(self):
    return self

  def next(self):
    # required for python 2.x compatibility
    idx = next(self.iter)
    return self.dataset.getImage(idx)

  def __next__(self):
    idx = next(self.iter)
    return self.dataset.getImage(idx)


class FolderImageDatasetReader(object):
  """
  Read images from a folder instead of a ROS bag file.
  Supports common image formats: jpg, png, bmp, etc.
  This makes calibration independent of ROS.
  """

  def __init__(self, folder, image_extensions=None, freq=None):
    """
    Initialize the folder-based dataset reader.

    Args:
      folder: Path to folder containing images
      image_extensions: List of image extensions to search for (default: ['.jpg', '.png', '.bmp'])
      freq: Optional frequency to subsample images [Hz]
    """
    self.folder = folder
    self.topic = folder  # Add topic attribute for compatibility
    self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    self.image_files = []
    self.timestamps = []

    if not os.path.isdir(folder):
      raise RuntimeError("Folder does not exist: {0}".format(folder))

    # Find all image files
    for ext in self.image_extensions:
      pattern = os.path.join(folder, '*' + ext)
      self.image_files.extend(sorted(glob.glob(pattern)))
      pattern = os.path.join(folder, '*' + ext.upper())
      self.image_files.extend(sorted(glob.glob(pattern)))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in self.image_files:
      if f not in seen:
        seen.add(f)
        unique_files.append(f)
    self.image_files = unique_files

    if len(self.image_files) == 0:
      raise RuntimeError("No image files found in folder: {0}".format(folder))

    # Generate timestamps based on file modification time or sequential numbering
    for idx, filepath in enumerate(self.image_files):
      try:
        # Try to use file modification time
        mtime = os.path.getmtime(filepath)
        timestamp = mtime
      except:
        # Fallback to sequential timestamps with 0.1s spacing
        timestamp = idx * 0.1
      self.timestamps.append(timestamp)

    # Create indices array
    self.indices = np.arange(len(self.image_files))

    # Apply frequency filtering if specified
    if freq:
      self.indices = self.truncateIndicesFromFreq(self.indices, freq)

    sm.logInfo("FolderImageDatasetReader: Found {0} images in folder: {1}".format(
        len(self.indices), folder))

  def truncateIndicesFromFreq(self, indices, freq):
    """Subsample images to match desired frequency"""
    if freq < 0.0:
      raise RuntimeError("Frequency {0} Hz is smaller 0".format(freq))

    # find the valid timestamps
    timestamp_last = -1
    valid_indices = []
    for idx in indices:
      timestamp = self.timestamps[idx]
      if timestamp_last < 0.0:
        timestamp_last = timestamp
        valid_indices.append(idx)
        continue
      if (timestamp - timestamp_last) >= 1.0 / freq:
        timestamp_last = timestamp
        valid_indices.append(idx)

    sm.logWarn(
      "FolderImageDatasetReader: truncated {0} / {1} images (frequency)".format(
          len(indices) - len(valid_indices), len(indices)))
    return valid_indices

  def __iter__(self):
    """Reset and return iterator"""
    return self.readDataset()

  def readDataset(self):
    """Return an iterator over the dataset"""
    return FolderImageDatasetReaderIterator(self, self.indices)

  def readDatasetShuffle(self):
    """Return an iterator over the shuffled dataset"""
    indices = self.indices.copy()
    np.random.shuffle(indices)
    return FolderImageDatasetReaderIterator(self, indices)

  def numImages(self):
    """Return the number of images in the dataset"""
    return len(self.indices)

  def getImage(self, idx):
    """
    Load and return an image with its timestamp.

    Args:
      idx: Index in the indices array

    Returns:
      Tuple of (timestamp, image_array)
    """
    actual_idx = self.indices[idx]
    filepath = self.image_files[actual_idx]
    timestamp_sec = self.timestamps[actual_idx]

    # Convert timestamp to aslam::Time format (seconds and nanoseconds)
    secs = int(timestamp_sec)
    nsecs = int((timestamp_sec - secs) * 1e9)
    timestamp = acv.Time(secs, nsecs)

    # Load image using OpenCV
    img_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img_data is None:
      raise RuntimeError("Could not read image file: {0}".format(filepath))

    return (timestamp, img_data)
