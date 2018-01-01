#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>

using namespace std;

class FeatureExtractor
{
    public:
        FeatureExtractor(vector<string> phrases, map<string, unsigned int> bag);
        virtual ~FeatureExtractor();

        void ProccessData();
        void ProccessData2File(string filename);
        vector<vector<int> > GetFeatures();
    protected:
    private:
        vector<vector<int> > _features;
        map<string, unsigned int> _bag;
        const vector<string> _phrases;

        vector<string> GetTokens(string phrase);
};

#endif // FEATUREEXTRACTOR_H
