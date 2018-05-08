import data_prepared as dp
import model
import sys, os

def main():
    print '---------------------------------------------'
    print 'Load image, black_white them, and resize them'
    print '---------------------------------------------'

    dp.ImagePreprocess.image_bw(old_image_folder='ori_organized')
    dp.ImagePreprocess.image_bw(old_image_folder='ori_disorganized')

    dp.ImagePreprocess.image_resize(old_image_folder='bw_ori_organized')
    dp.ImagePreprocess.image_resize(old_image_folder='bw_ori_disorganized')

    print '---------------------------------------------'
    print 'Convert images to numpy array'
    print '---------------------------------------------'

    good_re, bad_re = (
    dp.DataPreProcess.data_prepared('resize_bw_ori_organized',
                                    'resize_bw_ori_disorganized'))

    good_pre = dp.DataPreProcess.preprocess(good_re)
    bad_pre = dp.DataPreProcess.preprocess(bad_re)

    print '---------------------------------------------'
    print 'Train, eval, test sample split'
    print '---------------------------------------------'

    train_sample, train_label, eval_sample, eval_label = (
    dp.DataPreProcess.data_generate(good_pre, bad_pre, 150))

    print '---------------------------------------------'
    print 'Train model'
    print '---------------------------------------------'

    mema_classifier = model.train_model(train_sample, train_label)

    print '---------------------------------------------'
    print 'Evaluate model'
    print '---------------------------------------------'

    result = model.eval_model(eval_sample, eval_label, mema_classifier)

    print '---------------------------------------------'
    print 'Evaluation Accuracy'
    print '---------------------------------------------'
    print result
if __name__ == '__main__':
    main()
