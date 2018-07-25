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

    def load(self, add_gt=True, add_mask=False): # 로딩 함수 (mask는 알겠는데 ground truth가 뭐지?)
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
                self._use_absolute_file_name(img) # _use_absuloute_file_name 은 어디서 온거지?@@@@@
                if add_gt: # groung truth가 있다면
                    self._add_detection_gt(img, add_mask) # _add_detection_gt함수를 수행하라. 근데 이 함수 뭐지? @@@@@
            return imgs # 로딩함수의 리턴값은 이미지 객체이다.

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['id']]  # equivalent but faster than the above two lines

        # clean-up boxes
        valid_objs = []
        width = img['width']
        height = img['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = obj['bbox']
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do assume that (0.0, 0.0) is upper-left corner of the first pixel
            box = FloatBox(float(x1), float(y1),
                           float(x1 + w), float(y1 + h))
            box.clip_by_shape([height, width])
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and box.is_box() and box.area() >= 4:
                obj['bbox'] = [box.x1, box.y1, box.x2, box.y2]
                valid_objs.append(obj)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert obj['iscrowd'] == 1
                        obj['segmentation'] = None
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) < len(segs):
                            log_once("Image {} has invalid polygons!".format(img['file_name']), 'warn')

                        obj['segmentation'] = valid_segs

        # all geometrically-valid boxes are returned
        boxes = np.asarray([obj['bbox'] for obj in valid_objs], dtype='float32')  # (n, 4)
        cls = np.asarray([
            COCOMeta.category_id_to_class_id[obj['category_id']]
            for obj in valid_objs], dtype='int32')  # (n,)
        is_crowd = np.asarray([obj['iscrowd'] for obj in valid_objs], dtype='int8')

        # add the keys
        img['boxes'] = boxes        # nx4
        img['class'] = cls          # n, always >0
        img['is_crowd'] = is_crowd  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = [
                obj['segmentation'] for obj in valid_objs]

    def print_class_histogram(self, imgs):
        nr_class = len(COCOMeta.class_names)
        hist_bins = np.arange(nr_class + 1)

        # Histogram of ground-truth objects
        gt_hist = np.zeros((nr_class,), dtype=np.int)
        for entry in imgs:
            # filter crowd?
            gt_inds = np.where(
                (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
            gt_classes = entry['class'][gt_inds]
            gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
        data = [[COCOMeta.class_names[i], v] for i, v in enumerate(gt_hist)]
        data.append(['total', sum([x[1] for x in data])])
        table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
        logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))

    @staticmethod
    def load_many(basedir, names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.

        Returns the same format as :meth:`COCODetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            coco = COCODetection(basedir, n)
            ret.extend(coco.load(add_gt, add_mask=add_mask))
        return ret


if __name__ == '__main__':
    c = COCODetection(cfg.DATA.BASEDIR, 'train2014')
    gt_boxes = c.load(add_gt=True, add_mask=True)
    print("#Images:", len(gt_boxes))
    c.print_class_histogram(gt_boxes)
