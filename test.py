import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE_1 = 'results/SynthText_1.h5'
OUT_FILE_2 = 'results/SynthText_2.h5'

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print colorize(Color.RED,'Data not found and have problems downloading.',bold=True)
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    db['data'][dname].attrs['txt'] = res[i]['txt']


def main(viz=False):
  # open databases:
  print colorize(Color.BLUE,'getting data..',bold=True)
  db = get_data()
  print colorize(Color.BLUE,'\t-> done',bold=True)

  # open the output h5 file:
  out_db_1 = h5py.File(OUT_FILE_1,'w')
  out_db_1.create_group('/data')
  out_db_2 = h5py.File(OUT_FILE_2,'w')
  out_db_2.create_group('/data')

  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE_1, bold=True)
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE_2, bold=True)

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in xrange(start_idx,end_idx-1):
    imname_1 = imnames[i]
    imname_2 = imnames[i+1]
    try:
      # get the image:
      img_1 = Image.fromarray(db['image'][imname_1][:])
      img_2 = Image.fromarray(db['image'][imname_2][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth_1 = db['depth'][imname_1][:].T
      depth_1 = depth_1[:,:,1]
      depth_2 = db['depth'][imname_2][:].T
      depth_2 = depth_2[:,:,1]

      # get segmentation:
      seg_1 = db['seg'][imname_1][:].astype('float32')
      area_1 = db['seg'][imname_1].attrs['area']
      label_1 = db['seg'][imname_1].attrs['label']

      seg_2 = db['seg'][imname_2][:].astype('float32')
      area_2 = db['seg'][imname_2].attrs['area']
      label_2 = db['seg'][imname_2].attrs['label']


      # re-size uniformly:
      sz_1 = depth_1.shape[:2][::-1]
      img_1 = np.array(img_1.resize(sz_1,Image.ANTIALIAS))
      seg_1 = np.array(Image.fromarray(seg_1).resize(sz_1,Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res_1 = RV3.render_text(img_1,depth_1,seg_1,area_1,label_1,
                              ninstance=INSTANCE_PER_IMAGE,viz=viz)
      sz_2 = depth_2.shape[:2][::-1]
      img_2 = np.array(img_2.resize(sz_2,Image.ANTIALIAS))
      seg_2 = np.array(Image.fromarray(seg_2).resize(sz_2,Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i+1,end_idx-1), bold=True)
      res_2 = RV3.render_text(img_2,depth_2,seg_2,area_2,label_2,
                              ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res_1) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(str(start_idx)+"_"+imname_1,res_1,out_db_1)
      if len(res_2) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(str(start_idx)+"_"+imname_2,res_2,out_db_2)
      # visualize the output:
      if viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  db.close()
  out_db_1.close()
  out_db_2.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)
