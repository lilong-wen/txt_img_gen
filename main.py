import numpy as np
import h5py
import os, sys, traceback
import os
import yaml
from render_res import Renderer
from common import *
from PIL import Image

def get_data(data_path):

    '''
    data_path = os.path.join(config['DATA_PATH'], DB_FNAME)
    '''
    if not os.path.exists(data_path):
        print("data not exists")

    return h5py.File(data_path, 'r')

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    print(np.string_(res[i]['txt']))
    db['data'][dname].attrs['txt'] = np.string_(res[i]['txt'])



if __name__ == "__main__":

    with open('./config.yaml') as conf:
        config = yaml.load(conf)

    db_origin = get_data(os.path.join(config['DATA_PATH'], config['DB_FNAME']))

    out_db_1 = h5py.File(config['OUTPUT_FILE_1'], 'w')
    out_db_1.create_group('/data')
    out_db_2 = h5py.File(config['OUTPUT_FILE_2'], 'w')
    out_db_2.create_group('/data')

    imnames = sorted(db_origin['image'].keys())
    N_img_origin = len(imnames)
    NUM_IMG = config['NUM_IMG']

    if NUM_IMG < 0:
        NUM_IMG = N_img_origin

    start_idx, end_idx = 0, min(NUM_IMG, N_img_origin)
    Render = Renderer(config['DATA_PATH'], max_time=config['SECS_PER_IMG'])
    render_dict_1 = {}
    render_dict_2 = {}
    for i in range(start_idx, end_idx-1):

        img_name_1 = imnames[i]
        img_name_2 = imnames[i+1]

        try:
            # image 1

            img_1 = Image.fromarray(db_origin['image'][img_name_1][:])

            depth_1 = db_origin['depth'][img_name_1][:].T
            depth_1 = depth_1[:,:,1]

            seg_1 = db_origin['seg'][img_name_1][:].astype('float32')
            area_1 = db_origin['seg'][img_name_1].attrs['area']
            label_1 = db_origin['seg'][img_name_1].attrs['label']

            sz_1 = depth_1.shape[:2][::-1]
            img_1 = np.array(img_1.resize(sz_1,Image.ANTIALIAS))
            seg_1 = np.array(Image.fromarray(seg_1).resize(sz_1,Image.NEAREST))

            # render_dict_1 = {img: img_1, depth: depth_1, seg: seg_1, \
            #                  area: area_1, label: label_1}
            render_dict_1['img'] = img_1
            render_dict_1['depth'] = depth_1
            render_dict_1['seg'] = seg_1
            render_dict_1['area'] = area_1
            render_dict_1['label'] = label_1

            # print(colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
            # res_1 = Render.render_text(img_1, depth_1, seg_1, area_1, label_1, \
            #                          ninstance=INSTANCE_PER_IMAGE, \
            #                          viz=config['VIS'])
            #
            # if len(res) > 0:
            #
            #     add_res_to_db(img_name, res, out_db_1)

            # image 2
            img_2 = Image.fromarray(db_origin['image'][img_name_2][:])
            depth_2 = db_origin['depth'][img_name_2][:].T
            depth_2 = depth_2[:,:,1]

            seg_2 = db_origin['seg'][img_name_2][:].astype('float32')
            area_2 = db_origin['seg'][img_name_2].attrs['area']
            label_2 = db_origin['seg'][img_name_2].attrs['label']

            sz_2 = depth_2.shape[:2][::-1]
            img_2 = np.array(img_2.resize(sz_2,Image.ANTIALIAS))
            seg_2 = np.array(Image.fromarray(seg_2).resize(sz_2,Image.NEAREST))

            # render_dict_2 = {img: img_2, depth: depth_2, seg: seg_2, \
            #                  area: area_2, label: label_2}
            render_dict_2['img'] = img_2
            render_dict_2['depth'] = depth_2
            render_dict_2['seg'] = seg_2
            render_dict_2['area'] = area_2
            render_dict_2['label'] = label_2

            res = Render.render_text(render_dict_1, render_dict_2, \
                                     ninstance=config['INSTANCE_PER_IMAGE'], \
                                     viz=config['VIS'])

            res1, res2 = res

            if len(res1) > 0 and len(res2)>0:

                add_res_to_db(img_name_1, res1, out_db_1)
                add_res_to_db(img_name_2, res2, out_db_2)


        except:
            traceback.print_exc()
            continue

    db_origin.close()
    out_db_1.close()
