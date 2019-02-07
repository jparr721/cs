#ifndef BAYES_H_
#define BAYES_H_

#include <string>
#include <unordered_map>

namespace bayes {
  using frequency_map =
    std::unordered_map<std::string, int>;

  struct document {
    document(const std::string& words);

    int word_count(const std::string& word) const;

    const std::string topic;
    frequency_map frequencies_;
  };

  class Bayes {
    public:
      Bayes(const std::string& data_file);

      double fit();
    private:
      void classifier();
      void stem_document();
      void lemmatize_document();
  };
} // namespace bayes

#endif // BAYES_H_
