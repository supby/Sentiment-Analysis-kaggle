#include "BagWordsGen.h"

BagWordsGen::BagWordsGen(vector<string> input, unsigned int minWordLength, unsigned int minFrequency):
                            _inputStrings(input),
                            _minWordLength(minWordLength),
                            _minFrequency(minFrequency)
{
    //ctor
}

BagWordsGen::~BagWordsGen()
{
    //dtor
}

void BagWordsGen::GenerateBag()
{
    _bag.clear();

    for (vector<string>::const_iterator it = _inputStrings.begin(); it != _inputStrings.end(); ++it) {
        istringstream iss(*it);
        vector<string> tokens;
        copy(istream_iterator<string>(iss),
             istream_iterator<string>(),
             back_inserter<vector<string> >(tokens));

        for (vector<string>::const_iterator itt = tokens.begin(); itt != tokens.end(); ++itt) {
            string w = *itt;
            if(w.length() >= _minWordLength
                    && std::find_if(w.begin(), w.end(),
                            std::not1(std::ptr_fun((int(*)(int))std::isalpha))) == w.end())
                _bag[w]++;
        }
    }
    // filter by frequency
    if(_minFrequency > 0)
    {
        map <string,unsigned int> shadowBag(_bag);
        for (map <string,unsigned int>::iterator cur=shadowBag.begin();cur!=shadowBag.end();cur++)
            if((*cur).second < _minFrequency)
                _bag.erase((*cur).first);
    }

}

map<string, unsigned int> BagWordsGen::GetBag()
{
    return _bag;
}
