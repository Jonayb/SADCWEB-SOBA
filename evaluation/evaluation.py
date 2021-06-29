import statistics
import matplotlib.pyplot as plt
from scipy import stats

summarize_ontology = False;  # set to True if you want to summarize the 10 folds of the ontology
create_boxplot = False;  # set to True if you want to create a boxplot using the combined accuracies of all ontologies
perform_ttest = True;  # set to True if you want to perform Welch's t-test between two ontologies

if summarize_ontology:
    aspects_classified = [125.0, 128.0, 146.0, 151.0, 148.0, 140.0, 151.0, 143.0, 142.0, 137.0]
    ontology_accuracy = [0.896, 0.90625, 0.8767, 0.8940, 0.9257, 0.9143, 0.8940, 0.8951, 0.9155, 0.8759]
    backup_accuracy = [0.815385, 0.834646, 0.777778, 0.766990, 0.759615, 0.714286, 0.792079, 0.798165, 0.800000, 0.834783]
    print('Aspects classified: ', statistics.mean(aspects_classified), statistics.mean(aspects_classified)/255)
    print('Ontology accuracy: ', statistics.mean(ontology_accuracy))
    print('Backup accuracy: ', statistics.mean(backup_accuracy))

if create_boxplot or perform_ttest:
    ca_SOBA = [0.835294, 0.839216, 0.814961, 0.862205, 0.845238, 0.837302, 0.865079, 0.849206, 0.841270, 0.829365]
    ca_WEBSOBA = [0.866667, 0.850980, 0.830709, 0.870079, 0.861111, 0.845238, 0.876984, 0.853175, 0.865079, 0.869048]
    ca_BERT = [0.847059, 0.843137, 0.799213, 0.846457, 0.876984, 0.833333, 0.869048, 0.817460, 0.841270, 0.837302]
    ca_BERTPOST = [0.803922, 0.854902, 0.803150, 0.822835, 0.845238, 0.813492, 0.821429, 0.789683, 0.817460, 0.805556]
    ca_T5 = [0.854902, 0.850980, 0.826772, 0.866142, 0.861111, 0.841270, 0.884921, 0.821429, 0.849206, 0.853175]
    ca_sasobus = [0.815686, 0.847059, 0.814961, 0.838583, 0.829365, 0.829365, 0.876984, 0.801587, 0.849206, 0.825397]
    ca_roberta = [0.854902, 0.827451, 0.814961, 0.842520, 0.873016, 0.845238, 0.865079, 0.825397, 0.829365, 0.833333]
    ca_manual = [0.854902, 0.870588, 0.834646, 0.842520, 0.857143, 0.825397, 0.853175, 0.853175, 0.865079, 0.857143]

    # data matrix with accuracies of all ontologies
    data = [ca_manual, ca_SOBA, ca_sasobus, ca_WEBSOBA, ca_BERT, ca_BERTPOST, ca_roberta, ca_T5]

    if perform_ttest:
        first = data[0]  # first ontology to include in the t-test
        second = data[1]  # second ontology to include in the t-test
        print(stats.ttest_ind(first, second))

    if create_boxplot:
        fig1, ax1 = plt.subplots()
        ax1.boxplot(data)
        plt.xticks(fontsize=8)
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8],
                   ['Manual', 'SOBA', 'SASOBUS', 'WEB-SOBA', 'BERT', 'BERT-POST', 'RoBERTa', 'T5'])
        plt.savefig('img/boxplot.png')