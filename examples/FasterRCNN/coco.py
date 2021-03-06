# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np # numpy module
import os # os module
from termcolor import colored # 텍스트에 칼라를 주는 모듈
from tabulate import tabulate # 리스트 이런거는 표로 만들어주는 모듈
import tqdm # 어느정도 진행됬는지 GUI 모듈.

from tensorpack.utils import logger
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

from config import config as cfg


__all__ = ['COCODetection', 'COCOMeta'] # 이것은 무엇을 의미하는 코드인가요.@@@@@


class _COCOMeta(object): # 인풋으로 오브젝트를 받아온다. 아웃풋은 무언가 시작할때, 클래스가 그런거 만들어주는 것같다.@@@@@
    INSTANCE_TO_BASEDIR = {  # 인스턴드투베이스디렉토리 라는 이름으로 딕셔너리를 하나 만든다.
        'train2014': 'train2014', 
        'val2014': 'val2014',
        'valminusminival2014': 'val2014',
        'minival2014': 'val2014',
        'test2014': 'test2014'
    }

    def valid(self):  # 유효성을 판단해주는 그런 함수인가요? @@@@@
        return hasattr(self, 'cat_names') # 이 함수는 또 뭐지:

    def create(self, cat_ids, cat_names): # 생성해주는 함수이다. 리스트에 있는 아이디와 이름들 중에서 고양이 아이디와, 고양이 이름을 생성해준다.
        """
        cat_ids: list of ids
        cat_names: list of names
        """
        assert not self.valid() # assert 는 디버깅할때만 효과가 있고 릴리즈에서는 안쓴다. 의미는 valid하면 에러를 발생해줘! 라는 의미.
        assert len(cat_ids) == cfg.DATA.NUM_CATEGORY and len(cat_names) == cfg.DATA.NUM_CATEGORY # 고양이의 이름과 아이디가 cfg에 있는 데이터와 같지 않으면 에러가 발생한다.
        self.cat_names = cat_names # 고양이 이름 초기화
        self.class_names = ['BG'] + self.cat_names # BG를 붙혀서 클래스이름 초기화.

        # background has class id of 0
        self.category_id_to_class_id = { # 카테고리 아이디를 클래스 아이디로 dictionary 하나 만든다.
            v: i + 1 for i, v in enumerate(cat_ids)} # 여기서 i 는 0부터 시작하는 인덱스 v는 고양이 이름이다.
        self.class_id_to_category_id = { # 클래스 -> 카테고리 dictionary 하나 만든다.
            v: k for k, v in self.category_id_to_class_id.items()}
        cfg.DATA.CLASS_NAMES = self.class_names


COCOMeta = _COCOMeta() # 세션하나 만들어 주는 것인가?@@@@@


class COCODetection(object): # 코코디텍션이라는 클래스 만든다.
    def __init__(self, basedir, name): # 초기화 함수- 베이스 디렉토리와 이름이 필요하다.
        assert name in COCOMeta.INSTANCE_TO_BASEDIR.keys(), name # 인스턴스투베이스디렉토리 딕셔너리 안에 이름이 없으면 예외처리해준다. 있어야한다.
        self.name = name # 이름 초기화
        self._imgdir = os.path.realpath(os.path.join(  # 이미지가 있는 디렉토리와 관련된 라인
            basedir, COCOMeta.INSTANCE_TO_BASEDIR[name]))
        assert os.path.isdir(self._imgdir), self._imgdir # 이미지가 없으면  self._imgdir  로 예외를 처리해준다.
        annotation_file = os.path.join( # 파일 패스를 조인해준다. 무슨 파일 패스? @@@@@
            basedir, 'annotations/instances_{}.json'.format(name))
        assert os.path.isfile(annotation_file), annotation_file # 패스에 파일이 없으면 annotation_file로 예외처리

        from pycocotools.coco import COCO # cocodataset에 있는 coco 모듈을 쓴다.
        self.coco = COCO(annotation_file) # 파일을 연다.

        # initialize the meta
        cat_ids = self.coco.getCatIds() # coco 모듈의 getcatids()를 쓴다. 그래서 cat_ids 를 초기화한다.
        cat_names = [c['name'] for c in self.coco.loadCats(cat_ids)]  # cat_names 리스트를 coco 모듈에 의해 각각 초기화한다.
        if not COCOMeta.valid(): # 유효하지 않다면
            COCOMeta.create(cat_ids, cat_names) # 하나 생성한다.
        else: # 유효하다면
            assert COCOMeta.cat_names == cat_names # 이름이 같은지 예외처리 해준다.

        logger.info("Instances loaded from {}.".format(annotation_file)) # log 를 남겨준다.

    def load(self, add_gt=True, add_mask=False): # 로딩 함수 (mask는 알겠는데 ground truth가 뭐지?)@@@@@
        """
        Args: 
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        if add_mask: # 만약 마스크가 있다면
            assert add_gt # ground truth가 없으면 예외처리
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)): # timed_operation은 어디서 온거지?@@@@@
            img_ids = self.coco.getImgIds() # 코코를 이용한 이미지 아이디 초기화
            img_ids.sort() # 이미지 아이디 정렬
            # list of dict, each has keys: height,width,id,file_name 
            imgs = self.coco.loadImgs(img_ids) # 이미지 아이디를 인덱스로 한 이미지 로딩

            for img in tqdm.tqdm(imgs): # tqdm으로 진행상황을 보여준다.
                self._use_absolute_file_name(img) # _use_absuloute_file_name 은 어디서 온거지? 바로 아래에서 온다.
                if add_gt: # groung truth가 있다면
                    self._add_detection_gt(img, add_mask) # _add_detection_gt함수를 수행하라. 근데 이 함수 뭐지? 아래에서 온다.
            return imgs # 로딩함수의 리턴값은 이미지 객체이다.

    def _use_absolute_file_name(self, img): # 이미지값을 받아와서 이미지 이름을 파일의 패스로 지정해준다.
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name'] # 파일이 없으면 예외처리해준다.

    def _add_detection_gt(self, img, add_mask): # 디텍션을 위해 박스와 클래스와 is_crowd를 만든다. 이것인 ground truth인가?@@@@@
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['id']]  # equivalent but faster than the above two lines # id 값을 통해 이미지 객체 만든다.

        # clean-up boxes
        valid_objs = [] # 리스트 하나 만들어서
        width = img['width'] # 이미지 객체에 대한 width 값 초기화
        height = img['height'] # height 값 초기화
        for obj in objs: # 객체들 중에서
            if obj.get('ignore', 0) == 1: # ignore 값이 있는 딕셔너리가 있으면 뛰어넘는다. 보지 않는다.
                continue
            x1, y1, w, h = obj['bbox'] # 객체에서 bbox 정보를 초기화한다.
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do assume that (0.0, 0.0) is upper-left corner of the first pixel
            box = FloatBox(float(x1), float(y1), # float 박스를 만든다.(네모난 박스를 만든다.)
                           float(x1 + w), float(y1 + h))
            box.clip_by_shape([height, width]) # clip_by_shape함수가 뭐지?@@@@@
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and box.is_box() and box.area() >= 4: # 객체의 너비가 1보다 크고 박스가 있고 박스의 너비가 4이상이면
                obj['bbox'] = [box.x1, box.y1, box.x2, box.y2] # 객체의 bbox는 x1,x2,y1,y2로 지정해준다.
                valid_objs.append(obj) # 그리고 객체를 유효한 객체들의 리스트에 넣는다.

                if add_mask: # 그리고 여기서 마스크가 있으면(mask r cnn 일때를 말한다.)
                    segs = obj['segmentation']  # 객체의 segmentation 부분을 가지고 segs 라는 변수를 초기화한다.
                    if not isinstance(segs, list): # segs 라는 변수가 리스트가 아닐때,
                        assert obj['iscrowd'] == 1 # 객체에 iscrowd가 1이면 예외처리해준다.
                        obj['segmentation'] = None # 객체의 segmentation이 없다? @@@@@
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6] # segs라는 리스트에서 유효한 것들만 뽑는다.
                        if len(valid_segs) < len(segs): # 근데 segs들에서 유효한 segs들이 별로 없다면
                            log_once("Image {} has invalid polygons!".format(img['file_name']), 'warn') # 로그를 띄운다. 별로 없어서 warning이라고

                        obj['segmentation'] = valid_segs # 유효한 segs들은 객체의 segmentation에 다시 넣어준다.

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4) # 유효한 객체들의 bbox를 np를 통해 만든어 준다.
        cls = np.asarray([ # 유효한 객체들로 클래스를 만든어 cls라는 변수를 만들어 준다. 
            COCOMeta.category_id_to_class_id[obj['category_id']]
            for obj in valid_objs], dtype='int32')  # (n,) 
        is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8') # 유효한 객체에서 각 객체의 is_crowd 를 가지고 is_crowd라는 똑같은 이름의 변수를 초기화해준다.

        # add the keys
        img['boxes'] = boxes        # nx4  # 박스들을 이미지 객체의 박스에 넣는다.  여기서 boxes는 아마 유효한 것들의 boxes 일것이다.
        img['class'] = cls          # n, always >0   # 클래스들을 이미지 객체의 클래스에 넣는다.
        img['is_crowd'] = is_crowd  # n, # is_crowd를 이미지 객체의 그것에 넣는다.
        if add_mask: # 만약 마스크 rcnn 이라면
            # also required to be float32
            img['segmentation'] = [ # 세그멘테이션도 이와 동일한 맥락이다.
                obj['segmentation'] for obj in valid_objs]

    def print_class_histogram(self, imgs): # 이미지 객체 리스트를 가지고와서 클래스 히스토그램을 그린다.
        nr_class = len(COCOMeta.class_names)  # 클래스 이름의 길이를 nr_class라는 변수에 초기화해준다.
        hist_bins = np.arange(nr_class + 1) # nr_class 에 1일 더하여 hist_bins라는 np를 만든다.

        # Histogram of ground-truth objects
        gt_hist = np.zeros((nr_class,), dtype=np.int) # ground truth 히스토그램 변수를 만든다.
        for entry in imgs: # 이미지 리스트에서 entry 라는 이름으로,
            # filter crowd?
            gt_inds = np.where( # numpy의 where 을 통해서 클래스가 0보다 크고, is_crowd가 0인 것의 위치를 찾는다.)
                (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
            gt_classes = entry['class'][gt_inds] # 그래서 찾은 것을 gt_classes라는 것에 넣어준다.
            gt_hist += np.histogram(gt_classes, bins=hist_bins)[0] # np histogram이라는 것을 사용해서 gt_hist를 만들어 준다. 이것을 반복한다.
        data = [[COCOMeta.class_names[i], v] for i, v in enumerate(gt_hist)] # data 라는 곳에 유용한 것들을 모아놓는다. 결국 보여줄 것.
        data.append(['total', sum([x[1] for x in data])]) # 데이터에 total 과 data의 있는 값들을 더 추가한다.
        table = tabulate(data, headers=['class', '#box'], tablefmt='pipe') # 표를 만들어 준다.
        logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan')) # 로그를 띄워준다.

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):  # 여러개를 로드하고 합병해서 특정 인스턴스파일들을 뭉친다.
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`. # cocodetection.load 라는 것과 같은 방식으로 리턴한다.
        """
        if not isinstance(names, (list, tuple)): # 인스턴드가 list, tuple이 아니면, 이렇게 해석하는것 맞나?  @@@@@ 
            names = [names] # 이름들을 names에 초기화한다.
        ret = [] # 리스트 하나 만들고, 
        for n in names: # 이름들을 가지고,
            coco = COCODetection(basedir, n) # 코코 변수 하나를 만든다. 코코 변수는 무엇일까? @@@@@
            ret.extend(coco.load(add_gt, add_mask=add_mask)) # ret에 coco.load를 해서 가지고 온 것을 계속 넣어준다. 그리고 반환한다.
        return ret 


if __name__ == '__main__': # 메인 함수 시작
    c = COCODetection(cfg.DATA.BASEDIR, 'train2014') # 코코 디텍션에서 cfg라는 곳의 베이스폴더 이름과 train2014를 가지고 c라는 이름에 넣어준다.
    gt_boxes = c.load(add_gt=True, add_mask=True) # ground truth 박스도 c에서 넣어준다.
    print("#Images:", len(gt_boxes)) # gt_boxes의 길이를 출력
    c.print_class_histogram(gt_boxes) # 그림을 그려준다. gt_bexes의 길이이다.
