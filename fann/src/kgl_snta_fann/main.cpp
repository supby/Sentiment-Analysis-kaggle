#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>

#include "fann.h"
#include "floatfann.h"
#include "parallel_fann.h"
#include "src/parallel_fann.c"

#include "BagWordsGen.h"
#include "FeatureExtractor.h"

using namespace std;

struct TrainData
{
    vector<string> phrases;
    vector<int> sentiments;
};

vector<vector<int> > loadFeaturesFromFile(string filename)
{
    vector<vector<int> > ret;

    ifstream infile(filename.data());
    string line;
    while (std::getline(infile, line, '\n'))
    {
        vector<int> fVector;
        std::istringstream iss(line);
        int val;
        while(iss >> val)
            fVector.push_back(val);
        ret.push_back(fVector);
    }

    infile.close();

    return ret;
}

TrainData loadTrainData(string filename)
{
    ifstream infile(filename.data());

    vector<string> phrases;
    vector<int> sentiments;
    string line;
    int lineNum = 0;
    while (std::getline(infile, line, '\n'))
    {
        lineNum++;
        if(lineNum == 1)
            continue;

        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while(std::getline(iss, token, '\t'))
            tokens.push_back(token);

        phrases.push_back(tokens[2]);

        int sentVal = 0;
        std::istringstream ss(tokens[3]);
        ss >> sentVal;
        sentiments.push_back(sentVal);
    }

    infile.close();

    TrainData ret;
    ret.phrases = phrases;
    ret.sentiments = sentiments;

    return ret;
}

void prepareFeatures(vector<string> phrases, string bagWordsFilename, string featuresFilename)
{
    cout << "Generate Bag Words" << endl;

    BagWordsGen bagGen(phrases, 5, 30);
    bagGen.GenerateBag();
    map<string, unsigned int> bag = bagGen.GetBag();

    cout << "Done. Bag size is " << bag.size() << endl;

    ofstream outfile(bagWordsFilename.data());
    for (map <string,unsigned int>::const_iterator cur=bag.begin(); cur!=bag.end(); cur++)
    {
        outfile << (*cur).first << ":" << (*cur).second << endl;
    }
    outfile.close();

    cout << "Extract features" << endl;

    FeatureExtractor fa(phrases, bag);
    fa.ProccessData2File(featuresFilename);

    cout << "Done." << endl;
}

void fannTrain(string trainFilename, string outFilename, int num_threads)
{
    const unsigned int max_epochs = 5000;
    unsigned int num_layers = 3;
    unsigned int num_neurons_hidden = 500;
    unsigned int num_output = 5;
    unsigned int num_input = 3208;
    const float desired_error = (const float) 0.001;
    const unsigned int epochs_between_reports = 10;

    struct fann *ann;
    struct fann_train_data *train_data;
    //struct fann_train_data *test_data;

    //struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
//    unsigned int layers[4] = {num_input, num_neurons_hidden, num_output};
//    struct fann *ann = fann_create_standard_array(num_layers, layers);
//
//    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
//	fann_set_activation_function_output(ann, FANN_SIGMOID);
//
//	fann_train_on_file(ann, trainFilename.data(), max_epochs, 1, desired_error);

//    struct fann *ann = fann_create_standard(num_layers, num_input,
//        num_neurons_hidden, num_output);

    cout << "Creating network." << endl;

    train_data = fann_read_train_from_file(trainFilename.data());

	ann = fann_create_standard(num_layers,
					  train_data->num_input, num_neurons_hidden, train_data->num_output);

	cout << "Training network." << endl;

    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
	fann_set_learning_momentum(ann, 0.4f);

	//fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);
    for(unsigned int i = 1; i <= max_epochs; i++)
	{
		float error = num_threads > 1 ? fann_train_epoch_irpropm_parallel(ann, train_data, num_threads) : fann_train_epoch(ann, train_data);
		cout << "Epochs    " << i << ". Current error: " << error << endl;
    }

	cout << "Saving trained NET." << endl;

	fann_save(ann, outFilename.data());

//    fann_set_activation_function_hidden(ann, FANN_SIGMOID);
//	fann_set_activation_function_output(ann, FANN_SIGMOID);
//
//	for(int i = 1; i <= max_epochs; i++)
//	{
//		float error = fann_train_epoch(ann, data);
//		printf("Epochs     %8d. Current error: %.10f\n", i, error);
//	}
//

    cout << "Cleaning up." << endl;
	fann_destroy_train(train_data);
	//fann_destroy_train(test_data);
	fann_destroy(ann);
}

vector<float> normalizeFeaturesVector(vector<int> featuresVector, int maxVal, int minVal)
{
    vector<float> row;
    for(unsigned int j=0; j<featuresVector.size(); j++)
        row.push_back((float)(featuresVector[j] - minVal)/ (maxVal-minVal));
    return row;
}

vector<vector<float> > normalizeFeatures(vector<vector<int> > features)
{
    int minVal = 9999, maxVal = 0;
    for(unsigned int i=0; i<features.size(); i++)
    {
        for(unsigned int j=0; j<features[i].size(); j++)
        {
            if(features[i][j] < minVal)
                minVal = features[i][j];
            if(features[i][j] > maxVal)
                maxVal = features[i][j];
        }
    }

    vector<vector<float> > ret;
    for(unsigned int i=0; i<features.size(); i++)
        ret.push_back(normalizeFeaturesVector(features[i], maxVal, minVal));

    return ret;
}

void prepareFannTrainData2File(string fannTrDataFilename, vector<vector<int> > features,
                               vector<int> sentiments, vector<int> fannConfig)
{
    map<int, string> sentiment2output;
    sentiment2output[0] = "1 0 0 0 0";
    sentiment2output[1] = "0 1 0 0 0";
    sentiment2output[2] = "0 0 1 0 0";
    sentiment2output[3] = "0 0 0 1 0";
    sentiment2output[4] = "0 0 0 0 1";

    ofstream outfile(fannTrDataFilename.data());

    for(unsigned int i=0; i<fannConfig.size(); i++)
    {
        outfile << fannConfig[i] << " ";
    }
    outfile << endl;

    vector<vector<float> > normalizedFeatures = normalizeFeatures(features);

    for(unsigned int i=0; i<normalizedFeatures.size(); i++)
    {
        for(unsigned int j=0; j<normalizedFeatures[i].size(); j++)
        {
            outfile << normalizedFeatures[i][j] << " ";
        }
        outfile << endl << sentiment2output[sentiments[i]] << endl;
    }

    outfile.close();
}

bool value_comparer(map<string, unsigned int>::value_type &i1, map<string, unsigned int>::value_type &i2)
{
    return i1.second<i2.second;
}

void featuresFile2FANNfeaturesFile(string fFilename, string fannFFilename,
                               vector<string> phrases, vector<int> sentiments)
{
    cout << "Generate Bag Words" << endl;

    BagWordsGen bagGen(phrases, 5, 30);
    bagGen.GenerateBag();
    map<string, unsigned int> bag = bagGen.GetBag();

    cout << "Done. Bag size is " << bag.size() << endl;

    int maxVal = (*(map<string, unsigned int>::const_iterator)max_element(bag.begin(), bag.end(), value_comparer)).second;
//    int minVal = (*(map<string, int>::const_iterator)min_element(bag.begin(), bag.end(), value_comparer)).second;
    int minVal = 0;
    cout << "maxVal=" << maxVal << endl;
    cout << "minVal=" << minVal << endl;

    map<int, string> sentiment2output;
    sentiment2output[0] = "1 0 0 0 0";
    sentiment2output[1] = "0 1 0 0 0";
    sentiment2output[2] = "0 0 1 0 0";
    sentiment2output[3] = "0 0 0 1 0";
    sentiment2output[4] = "0 0 0 0 1";

    ifstream infile(fFilename.data());
    ofstream outfile(fannFFilename.data());

    vector<int> fannConfig;
    fannConfig.push_back(156000);
    fannConfig.push_back(bag.size());
    fannConfig.push_back(5);

    for(unsigned int i=0; i<fannConfig.size(); i++)
        outfile << fannConfig[i] << " ";
    outfile << endl;

    cout << "Convert to FANN features" << endl;

    string line;
    int i = 0;
    while (std::getline(infile, line, '\n'))
    {
        vector<int> fVector;
        std::istringstream iss(line);
        int val;
        while(iss >> val)
            fVector.push_back(val);

        vector<float> normalizedVector = normalizeFeaturesVector(fVector, maxVal, minVal);

        for(unsigned int j=0; j<normalizedVector.size(); j++)
            outfile << normalizedVector[j] << " ";
        outfile << endl << sentiment2output[sentiments[i++]] << endl;

        cout << ".";
    }

    outfile.close();
    infile.close();

    cout << endl;
}

int main(int argc, char *argv[])
{
//    cout << "Loading train data ..." << endl;
//
//    TrainData trData = loadTrainData("D:\\MyProjects\\kaggl_sent\\data\\train.tsv");
//
//    vector<string> phrases = trData.phrases;
//    vector<int> sentiments = trData.sentiments;
//
//    cout << "Done." << endl;

//    prepareFeatures(phrases, "D:\\MyProjects\\kaggl_sent\\data\\bag_words.txt",
//                    "D:\\MyProjects\\kaggl_sent\\data\\train_features.txt");
//
//    return 0;

    //vector<vector<int> > features = loadFeaturesFromFile("E:\\dropbox\\Dropbox\\prj\\kaggle\\Sentiment Analysis\\data\\train_features.txt");


//    vector<int> fannConfig;
//    fannConfig.push_back(features[0].size());
//    fannConfig.push_back(500);
//    fannConfig.push_back(5);

    //cout << "Preparing data for FANN to file..." << endl;

    //prepareFannTrainData2File("E:\\dropbox\\Dropbox\\prj\\kaggle\\Sentiment Analysis\\data\\fann_sentiments.train",
    //features, sentiments, fannConfig);

    //cout << "Done." << endl;


//    featuresFile2FANNfeaturesFile("D:\\MyProjects\\kaggl_sent\\data\\train_features.txt",
//                              "D:\\MyProjects\\kaggl_sent\\data\\fann_sentiments.train",
//                              phrases, sentiments);


    string fannTrainFilename = "../../../data/fann_sentiments.10k.train";
    string  fannTrainOutFilename = "../../../data/fann_sentiments_10k_iter5000_cfg_3208_500_5.net";
    int num_threads = 4;
    if(argc >= 3)
    {
        fannTrainFilename = argv[1];
        fannTrainOutFilename = argv[2];
    }
    if(argc >= 4)
        std::stringstream(argv[3]) >> num_threads;

    fannTrain(fannTrainFilename, fannTrainOutFilename, num_threads);

    return 0;
}
