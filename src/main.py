import data_prepared as dp
import model
import sys, os

def main():
    mode = sys.argv[1]
    if mode == 'predict':
        print '---------------------------------------------'
        print 'Predicting new images'
        print '---------------------------------------------'

        pred_array, pred_file = dp.DataPreProcess.predict_prep('predict')
        pred_pre = dp.DataPreProcess.preprocess(pred_array)
        pred_result = model.pred_model(pred_pre)

        model.prediction(pred_result, pred_file)

    else:

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

        good_re = dp.DataPreProcess.data_prepared('resize_bw_ori_organized')
        bad_re = dp.DataPreProcess.data_prepared('resize_bw_ori_disorganized')

        good_pre = dp.DataPreProcess.preprocess(good_re)
        bad_pre = dp.DataPreProcess.preprocess(bad_re)

        print '---------------------------------------------'
        print 'Train, eval, test sample split'
        print '---------------------------------------------'

        train_sample, train_label, eval_sample, eval_label = (
        dp.DataPreProcess.data_generate(good_pre, bad_pre))

        if mode == 'train':
            print '---------------------------------------------'
            print 'Train model'
            print '---------------------------------------------'

            model.train_model(train_sample, train_label)

        if mode == 'eval':
            print '---------------------------------------------'
            print 'Evaluate model'
            print '---------------------------------------------'
            print '---------------------------------------------'
            print 'Evaluation Accuracy'
            print '---------------------------------------------'
            print model.eval_model(eval_sample, eval_label)

if __name__ == '__main__':
    main()
