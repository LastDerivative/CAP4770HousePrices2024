Need files: train.csv, test.csv, sample_submission
Will output the predictions made by the models ran in a format submittable to Kaggle

FullPipeline.py by default runs the GB model and the Neural Network Model with evaluation on both.
Total run time should be around: 10 minutes

Able to specify which models to run by specifying a combination of the following --run_gb --run_nn --run_stacked --run_all

i.e: FullPipeline.py --run_nn will only run the nn model with evaluation
	MSE scores for each fold: [6.63152554e+08 4.91717313e+08 4.91675384e+08 7.76904957e+08
 	6.05079971e+08]
	RMSE scores for each fold: [25751.74855413 22174.6998479  22173.75438609 27873.01484767
 	24598.37333298]
	Average MSE: 605706035.5682492
	Average RMSE: 24514.31819375379
	Average Sale Price: $180932.92
	Average RMSE: $24514.32
	RMSE as Percentage of Average Sale Price: 13.55%
	Baseline RMSE: $79467.79
	Model RMSE is better than baseline.
	Model Evaluation started at: 2024-07-30 22:05:50
	Model Evaluation finished at: 2024-07-30 22:13:17
	Total Evaluation execution time: 0:07:26.727285
	Logarithmic RMSE: 0.4167785842875396


Be default, evaluation of the stacked model is not ran do to the time it will take: Total run time of about 50 minutes

Able to specify stacked Evaluation when running the stacked model or all models with --evaluate_stacked 
i.e: FullPipeline.py --run_all --evaluate_stacked or FullPipeline.py --run_stacked --evaluate_stacked

Results of stacked Evaluation with outlier removal and aggressive preprocessing:
		Total Time: 50 minutes

		Total Training execution time: 0:10:24.215602
		Total Evaluation execution time: 0:40:07.320574

		MSE scores for each fold: [4.57695389e+08 4.78728189e+08 4.27827539e+08 6.84540074e+08
		 4.58626083e+08]
		RMSE scores for each fold: [21393.8165989  21879.85806529 20683.99232721 26163.71675458
		 21415.55703462]
		Average MSE: 501483454.7467955
		Average RMSE: 22307.38815612143
		Average Sale Price: $180932.92
		Average RMSE: $22307.39
		RMSE as Percentage of Average Sale Price: 12.33%
		Baseline RMSE: $79467.79
		Model RMSE is better than baseline.


		KAGGLE: .12942

Results of stacked model using old Pre-processing with not Evaluation:
		Total Training execution time: 0:11:44.544249
		Logarithmic RMSE: 0.3896247474797472
	
		KAGGLE: .0.12512

Be default, evaluation of the Neural Model is not ran do to the time it will take: Total run time of about 50 minutes


