# FlakyTestCategorization
This project the the realization of FlaKat which is the framework developed for Shizhe Lin Master Thesis "FlaKat: A Machine Learning-Based Categorization Framework for Flaky Tests"

The parsing of flaky tests only need to be executed once by running extract.py. The extrcted-all-projects.csv under main/data/input already contains the parsed flaky tests and vector.csv contains the results of runing code2vec (this step is not yet fully automated). Once these input data are ready, FlaKat is ready to go.

After running main.py, 6 digit code can be entered to select various options supported
1. embedding to use 1-doc2vec 2-code2vec 3-tfidf
2. sampling 0-none 1-TL 2-SMOTE 3-TL&SMOTE 4-SMOTE&TL
3. reduction 1-PCA 2-LDA 3-Isomap 4-t-SNE 5-UMAP
4. visualization of reduced embedding 0-none 1-2d 2-3d
5. classifier 1-KNN 2-SVM 3-RF 4-RF with tuning 5-GBDT 6-GBDT with tuning
6. result 0-csv 1-no log

One example would be: 342030

Configuration of specific components can be adjusted in main.py and the output including accuracy measurement and visualizations can be found in main/data/output.

There are few other jupyter notebook file that experiments the other aspects of the thesis
proveFDC.ipynb - proving the value of FDC 
teguchi.ipynb - finding most impactful RF hyperparameter by Teguchi array.
vector.ipynb - finding CVR representation of java source code
bayesianOptimization.ipynb - more detailed exploration on applying BO, simplified version is incorporated in FlaKat
