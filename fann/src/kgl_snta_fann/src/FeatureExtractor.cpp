#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(vector<string> phrases, map<string, unsigned int> bag) : _phrases(phrases), _bag(bag)
{
    //ctor
}

FeatureExtractor::~FeatureExtractor()
{
    //dtor
}

vector<vector<int> > FeatureExtractor::GetFeatures()
{
    return _features;
}

void FeatureExtractor::ProccessData()
{
    for (vector<string>::const_iterator it = _phrases.begin(); it != _phrases.end(); ++it)
    {
        vector<string> tokens = GetTokens(*it);
        vector<int> featuresVector;
        for (map <string,unsigned int>::const_iterator cur=_bag.begin();cur!=_bag.end();cur++)
        {
            vector<string>::iterator fit = std::find(tokens.begin(), tokens.end(), (*cur).first);
            featuresVector.push_back(fit == tokens.end() ? 0 : (*cur).second);
        }
        _features.push_back(featuresVector);
    }
}

void FeatureExtractor::ProccessData2File(string filename)
{
    ofstream outfile(filename.data());

    for (vector<string>::const_iterator it = _phrases.begin(); it != _phrases.end(); ++it)
    {
        vector<string> tokens = GetTokens(*it);
        for (map <string,unsigned int>::const_iterator cur=_bag.begin();cur!=_bag.end();cur++)
        {
            vector<string>::iterator fit = std::find(tokens.begin(), tokens.end(), (*cur).first);
            outfile << (fit == tokens.end() ? 0 : (*cur).second) << " ";
        }
        outfile << endl;
    }

    outfile.close();
}

vector<string> FeatureExtractor::GetTokens(string phrase)
{
    istringstream iss(phrase);
    vector<string> tokens;

    string word;
    while(iss >> word)
        tokens.push_back(word);

    return tokens;
}
