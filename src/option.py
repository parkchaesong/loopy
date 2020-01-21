import easydict

options = easydict.EasyDict({
        "iou_threshold": 0.98,
        "size_threshold": 32,
        "confidence_threshold": 0.3,
        "n_interpolation": 11,
        "no_interpolation": False,  # just use pascal voc 2007
        "ignore": None,
        "set_class_iou": None,  # 특정 클래스에 대해 특정 iou 설정, ex: person 0.7
        "quiet": True
    })


'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
