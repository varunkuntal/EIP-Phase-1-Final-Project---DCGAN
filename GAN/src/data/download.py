"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Downloads the following:
- Celeb-A dataset
"""

from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import requests
import subprocess
from tqdm import tqdm
from six.moves import urllib


parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('datasets', metavar='N', type=str, nargs='+', choices=['celebA', 'lsun', 'mnist'],
           help='name of dataset to download [celebA, lsun, mnist]')

def download_file_from_google_drive(fileid, path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': fileid}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': fileid, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, path):
    CHUNK_SIZE = 32768

    with open(path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download(url, dirpath):
  filename = url.split('/')[-1]
  filepath = os.path.join(dirpath, filename)
  u = urllib.request.urlopen(url)
  f = open(filepath, 'wb')
  filesize = int(u.headers["Content-Length"])
  print("Downloading: %s Bytes: %s" % (filename, filesize))

  downloaded = 0
  block_sz = 8192
  status_width = 70
  while True:
    buf = u.read(block_sz)
    if not buf:
      print('')
      break
    else:
      print('', end='\r')
    downloaded += len(buf)
    f.write(buf)
    status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
      ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
    print(status, end='')
    sys.stdout.flush()
  f.close()
  return filepath

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={ 'id': id }, stream=True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
  total_size = int(response.headers.get('content-length', 0))
  with open(destination, "wb") as f:
    for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
              unit='B', unit_scale=True, desc=destination):
      if chunk: # filter out keep-alive new chunks
        f.write(chunk)

def unzip(filepath):
  print("Extracting: " + filepath)
  dirpath = os.path.dirname(filepath)
  with zipfile.ZipFile(filepath) as zf:
    zf.extractall(dirpath)
  os.remove(filepath)

def download_celeb_a(dirpath):
  data_dir = 'img_align_celeba'
  if os.path.exists(os.path.join(dirpath, data_dir)):
    print('Found Celeb-A - skip')
    return

  filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
  save_path = os.path.join(dirpath, filename)

  if os.path.exists(save_path):
    print('[*] {} already exists'.format(save_path))
  else:
    download_file_from_google_drive(drive_id, save_path)

  zip_dir = ''
  with zipfile.ZipFile(save_path) as zf:
    zip_dir = zf.namelist()[0]
    zf.extractall(dirpath)
  os.remove(save_path)
  os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))

def _list_categories(tag):
  url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
  f = urllib.request.urlopen(url)
  return json.loads(f.read())

if __name__ == '__main__':
  args = parser.parse_args()
  # id and path
  readme_ids = [
     '0B7EVK8r0v71pOXBhSUdJWU1MYUk']
  readme_paths = [
        '/content/DeepLearningImplementations/GAN/data/raw/README.txt']

  annotation_ids = [
        '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
        '0B7EVK8r0v71pblRyaVFSWGxPY0U',
        '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
        '0B7EVK8r0v71pTzJIdlJWdHczRlU']
  annotation_paths = [
        '/content/DeepLearningImplementations/GAN/data/raw/Anno/list_bbox_celeba.txt',
        '/content/DeepLearningImplementations/GAN/data/raw/Anno/list_attr_celeba.txt',
        '/content/DeepLearningImplementations/GAN/data/raw/Anno/list_landmarks_align_celeba.txt',
        '/content/DeepLearningImplementations/GAN/data/raw/Anno/list_landmarks_celeba.txt']

  eval_ids = [
        '0B7EVK8r0v71pY0NSMzRuSXJEVkk']
  eval_paths = [
        '/content/DeepLearningImplementations/GAN/data/raw/Eval/list_eval_partition.txt']
    
  ids = readme_ids + annotation_ids + eval_ids 

  paths = readme_paths + annotation_paths + eval_paths
    # directory
  try:
      root = os.path.join(sys.argv[1], '/content/DeepLearningImplementations/GAN/data/raw')
  except:
      root = '/content/DeepLearningImplementations/GAN/data/raw/'
    
  Anno = os.path.join(root, '/content/DeepLearningImplementations/GAN/data/raw/Anno')
  Eval = os.path.join(root, '/content/DeepLearningImplementations/GAN/data/raw/Eval')

  if not os.path.exists(Anno):
      os.makedirs(Anno)

  if not os.path.exists(Eval):
      os.makedirs(Eval)
    

  if any(name in args.datasets for name in ['CelebA', 'celebA', 'celebA']):
      download_celeb_a('/content/DeepLearningImplementations/GAN/data/raw/')

  for i, (fileid, path) in enumerate(zip(ids, paths)):
        print('{}/{} downloading {}'.format(i + 1, len(ids), path))
        path = os.path.join(root, path)
        if not os.path.exists(path):
            download_file_from_google_drive(fileid, path)

