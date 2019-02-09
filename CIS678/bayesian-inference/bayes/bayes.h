#ifndef BAYES_H_
#define BAYES_H_

#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bayes {
  using frequency_map =
    std::unordered_map<std::string_view, int>;

  struct document {
    document(const std::string& doc_path);

    void load_document(const std::string& doc_path);
    void stem_document();
    void lemmatize_document();

    int word_count(const std::string& word) const;

    bool ends_with(const std::string& word, const std::string& suffix);

    std::vector<std::string> split(std::string line);

    std::vector<std::string> lines_;
    frequency_map topic_frequencies_;
    frequency_map word_frequencies_;
    const std::array<std::string, 3> suffixes{{"ed", "ing", "'s"}};
  };

  class Bayes {
    public:
      Bayes(const std::string& data_file);

      double fit();
    private:
      void classifier();
  };
} // namespace bayes

#endif // BAYES_H_
