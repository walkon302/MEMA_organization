import data_prepared as dp
import model
import sys, os

def main():
    mode = sys.argv[1]
    if mode == 'predict':
        output_name = sys.argv[2]
        print '---------------------------------------------'
        print 'Predicting new images'
        print '---------------------------------------------'

        pred_array, pred_file = dp.CNNDataPreProcess.predict_prep('predict')
        pred_pre = dp.CNNDataPreProcess.cnn_preprocess(pred_array)
        pred_result = model.pred_model(pred_pre)

        model.prediction(pred_result, pred_file, output_name)

    else:

        print '---------------------------------------------'
        print 'Load image, black_white them, and resize them'
        print '---------------------------------------------'

        if mode == 'train':
            training_step = int(sys.argv[2])

            train_sample, train_label = (
            dp.CNNDataPreProcess.train_eval_prep('train_organized',
                                              'train_disorganized',
                                              'train')
            )

            print '---------------------------------------------'
            print 'Train model'
            print '---------------------------------------------'

            model.train_model(train_sample, train_label, training_step)

        if mode == 'eval':

            eval_sample, eval_label = (
            dp.CNNDataPreProcess.train_eval_prep('eval_organized',
                                              'eval_disorganized',
                                              'eval')
            )

            print '---------------------------------------------'
            print 'Evaluate model'
            print '---------------------------------------------'
            print '---------------------------------------------'
            print 'Evaluation Accuracy'
            print '---------------------------------------------'

            print model.eval_model(eval_sample, eval_label)

if __name__ == '__main__':
    main()
