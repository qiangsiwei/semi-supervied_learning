This is the code of semi-supervised cost-sensitive boosting algorithms.  

This is the extension for AdaBoost in scikit-learn.  

http://scikit-learn.org/stable/modules/ensemble.html#adaboost  

基于UCI数据集进行了测试，实验结果如图。

其中R为代价矩阵的代价系数，包含以下算法：

*   DT：决策树

*   MlAda_DT：Adaboost

*   SSMAB_DT：半监督Adaboost

*   MlAda_DT_CS：代价敏感Adaboost

*   SSMAB_DT_CS：半监督代价敏感Adaboost

评价包含两个部分，即准确率和总代价，可见，SSMAB_DT_CS优于其他算法。

![Alt Text](https://raw.githubusercontent.com/qiangsiwei/semi-supervied_learning/master/result.png)

