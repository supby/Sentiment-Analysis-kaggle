#ifndef BAGWORDSGEN_H
#define BAGWORDSGEN_H

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cctype>
#include <functional>

using namespace std;

class BagWordsGen
{
    public:
        BagWordsGen(vector<string> input, unsigned int minWordLength, unsigned int minFrequency);
        virtual ~BagWordsGen();

        void GenerateBag();
        map<string, unsigned int> GetBag();
    protected:
    private:
        const vector<string> _inputStrings;
        map<string, unsigned int> _bag;
        unsigned int _minWordLength;
        unsigned int _minFrequency;
};

#endif // BAGWORDSGEN_H
