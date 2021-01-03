/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <queue>
#include <utility>
#include <stdexcept>
#include "args.h"
#include "autotune.h"
#include "fasttext.h"

using namespace fasttext;

void printUsage() {
  std::cerr
      << "usage: fasttext <command> <args>\n\n"
      << "The commands supported by fasttext are:\n\n"
      << "  supervised              train a supervised classifier\n"
      << "  quantize                quantize a model to reduce the memory "
         "usage\n"
      << "  test                    evaluate a supervised classifier\n"
      << "  test-label              print labels with precision and recall "
         "scores\n"
      << "  predict                 predict most likely labels\n"
      << "  predict-prob            predict most likely labels with "
         "probabilities\n"
      << "  skipgram                train a skipgram model\n"
      << "  cbow                    train a cbow model\n"
      << "  print-word-vectors      print word vectors given a trained model\n"
      << "  print-sentence-vectors  print sentence vectors given a trained "
         "model\n"
      << "  print-ngrams            print ngrams given a trained model and "
         "word\n"
      << "  nn                      query for nearest neighbors\n"
      << "  analogies               query for analogies\n"
      << "  dump                    dump arguments,dictionary,input/output "
         "vectors\n"
      << std::endl;
}

void printQuantizeUsage() {
  std::cerr << "usage: fasttext quantize <args>" << std::endl;
}

void printTestUsage() {
  std::cerr
      << "usage: fasttext test <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printPredictUsage() {
  std::cerr
      << "usage: fasttext predict[-prob] <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename (if -, read from stdin)\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printTestLabelUsage() {
  std::cerr
      << "usage: fasttext test-label <model> <test-data> [<k>] [<th>]\n\n"
      << "  <model>      model filename\n"
      << "  <test-data>  test data filename\n"
      << "  <k>          (optional; 1 by default) predict top k labels\n"
      << "  <th>         (optional; 0.0 by default) probability threshold\n"
      << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr << "usage: fasttext print-word-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr << "usage: fasttext print-sentence-vectors <model>\n\n"
            << "  <model>      model filename\n"
            << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr << "usage: fasttext print-ngrams <model> <word>\n\n"
            << "  <model>      model filename\n"
            << "  <word>       word to print\n"
            << std::endl;
}

void quantize(const std::vector<std::string>& args) {
  Args a = Args();
  if (args.size() < 3) {
    printQuantizeUsage();
    a.printHelp();
    exit(EXIT_FAILURE);
  }
  a.parseArgs(args);
  FastText fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a.output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel(a.output + ".ftz");
  exit(0);
}

void printNNUsage() {
  std::cout << "usage: fasttext nn <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printAnalogiesUsage() {
  std::cout << "usage: fasttext analogies <model> <k>\n\n"
            << "  <model>      model filename\n"
            << "  <k>          (optional; 10 by default) predict top k labels\n"
            << std::endl;
}

void printDumpUsage() {
  std::cout << "usage: fasttext dump <model> <option>\n\n"
            << "  <model>      model filename\n"
            << "  <option>     option from args,dict,input,output" << std::endl;
}

void test(const std::vector<std::string>& args) {
  bool perLabel = args[1] == "test-label";

  if (args.size() < 4 || args.size() > 6) {
    perLabel ? printTestLabelUsage() : printTestUsage();
    exit(EXIT_FAILURE);
  }

  const auto& model = args[2];
  const auto& input = args[3];
  int32_t k = args.size() > 4 ? std::stoi(args[4]) : 1;
  real threshold = args.size() > 5 ? std::stof(args[5]) : 0.0;

  FastText fasttext;
  fasttext.loadModel(model);

  Meter meter(false);

  if (input == "-") {
    fasttext.test(std::cin, k, threshold, meter);
  } else {
    std::ifstream ifs(input);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k, threshold, meter);
  }

  if (perLabel) {
    std::cout << std::fixed << std::setprecision(6);
    auto writeMetric = [](const std::string& name, double value) {
      std::cout << name << " : ";
      if (std::isfinite(value)) {
        std::cout << value;
      } else {
        std::cout << "--------";
      }
      std::cout << "  ";
    };

    std::shared_ptr<const Dictionary> dict = fasttext.getDictionary();
    for (int32_t labelId = 0; labelId < dict->nlabels(); labelId++) {
      writeMetric("F1-Score", meter.f1Score(labelId));
      writeMetric("Precision", meter.precision(labelId));
      writeMetric("Recall", meter.recall(labelId));
      std::cout << " " << dict->getLabel(labelId) << std::endl;
    }
  }
  meter.writeGeneralMetrics(std::cout, k);

  exit(0);
}

void printPredictions(
    const std::vector<std::pair<real, std::string>>& predictions,
    bool printProb,
    bool multiline) {
  bool first = true;
  for (const auto& prediction : predictions) {
    if (!first && !multiline) {
      std::cout << " ";
    }
    first = false;
    std::cout << prediction.second;
    if (printProb) {
      std::cout << " " << prediction.first;
    }
    if (multiline) {
      std::cout << std::endl;
    }
  }
  if (!multiline) {
    std::cout << std::endl;
  }
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 6) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  real threshold = 0.0;
  if (args.size() > 4) {
    k = std::stoi(args[4]);
    if (args.size() == 6) {
      threshold = std::stof(args[5]);
    }
  }

  bool printProb = args[1] == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::ifstream ifs;
  std::string infile(args[3]);
  bool inputIsStdIn = infile == "-";
  if (!inputIsStdIn) {
    ifs.open(infile);
    if (!inputIsStdIn && !ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  std::istream& in = inputIsStdIn ? std::cin : ifs;
  std::vector<std::pair<real, std::string>> predictions;
  while (fasttext.predictLine(in, predictions, k, threshold)) {
    printPredictions(predictions, printProb, false);
  }
  if (ifs.is_open()) {
    ifs.close();
  }

  exit(0);
}

void printWordVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string word;
  Vector vec(fasttext.getDimension());
  while (std::cin >> word) {
    fasttext.getWordVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
  exit(0);
}

void printSentenceVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  Vector svec(fasttext.getDimension());
  while (std::cin.peek() != EOF) {
    fasttext.getSentenceVector(std::cin, svec);
    // Don't print sentence
    std::cout << svec << std::endl;
  }
  exit(0);
}

void printNgrams(const std::vector<std::string> args) {
  if (args.size() != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string word(args[3]);
  std::vector<std::pair<std::string, Vector>> ngramVectors =
      fasttext.getNgramVectors(word);

  for (const auto& ngramVector : ngramVectors) {
    std::cout << ngramVector.first << " " << ngramVector.second << std::endl;
  }

  exit(0);
}

void nn(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 20;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printNNUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string prompt("Query word? ");
  std::cout << prompt;

  std::string queryWord;
  while (std::cin >> queryWord) {
    printPredictions(fasttext.getNN(queryWord, k), true, true);
    std::cout << prompt;
  }
  exit(0);
}

void mrr(const std::vector<std::string> args) {
    int32_t k;
    if (args.size() == 5) {
        k = 10;
    } else if (args.size() == 6) {
        k = std::stoi(args[5]);
    } else {
        exit(EXIT_FAILURE);
    }

    FastText fasttext;
    fasttext.loadModel(std::string(args[2]));
    FastText ft_moe;
    ft_moe.loadModel(std::string(args[3]));

    std::ifstream ifs(args[4]);
    if (!ifs.is_open()) {
        throw std::invalid_argument(args[4] + "cannot be opened for testing MRR");
    }
    std::string we;
    std::string wm;
    std::string temp;
    std::vector<std::string> all_words;
    std::vector<std::pair<std::string, std::string>> temp_pairs;
    int32_t total = 0;
    real score = 0;
    real score_moe = 0;

    int32_t count = 0;

    while (ifs >> temp) {
        all_words.push_back(temp);
    }
    temp = "";
    int32_t ind = 0;
    while (ind < all_words.size()) {
        if (all_words[ind].substr(0, 1).compare("$") == 0) {
            temp = all_words[ind].substr(1, all_words[ind].length());
        } else {
            std::pair<std::string, std::string> p = std::make_pair(temp, all_words[ind]);
            temp_pairs.push_back(p);
            count++;
            std::cerr << "\rRead in : " << count << " misspelled words" << std::flush;
        }
        ind++;
    }

    std::cerr << std::endl;

    while (total < count) {
        we = temp_pairs[total].first;
        wm = temp_pairs[total].second;
        std::vector<std::pair<real, std::string>> predictions = fasttext.getNN(wm, k);
        std::vector<std::pair<real, std::string>> pd = ft_moe.getNN(wm, k);
        real rank = 1;
        real rank_moe = 1;
        for (const auto& pred : predictions) {
            if (we.compare(pred.second) != 0) {
                rank++;
            }
            else {
                break;
            }
        }
        //std::cerr << "normal rank : " << rank << std::endl;
        if (rank <= 10) {
            score += 1 / rank;
        }

        for (const auto& pred : pd) {
            if (we.compare(pred.second) != 0) {
                rank_moe++;
            }
            else {
                break;
            }
        }
        //std::cerr << "MOE rank : " << rank_moe << std::endl;
        if (rank_moe <= 10) {
            score_moe += 1 / rank_moe;
        }
        total++;
        std::cerr << "\rProcessed : " << total << " / " << count << std::flush;
        //std::cerr << "\rProcessed : " << std::setprecision(4) << (double)(total / 600000) << "%" << std::flush;
    }
    std::cerr << "\nTotal MRR of baseline is : " << score / total << std::endl;
    std::cerr << "\nTotal MRR of MOE is : " << score_moe / total << std::endl;

    exit(0);
}

void analogies(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printAnalogiesUsage();
    exit(EXIT_FAILURE);
  }
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  FastText fasttext;
  std::string model(args[2]);
  std::cout << "Loading model " << model << std::endl;
  fasttext.loadModel(model);

  std::string prompt("Query triplet (A - B + C)? ");
  std::string wordA, wordB, wordC;
  std::cout << prompt;
  while (true) {
    std::cin >> wordA;
    std::cin >> wordB;
    std::cin >> wordC;
    printPredictions(fasttext.getAnalogies(k, wordA, wordB, wordC), true, true);

    std::cout << prompt;
  }
  exit(0);
}

void train(const std::vector<std::string> args) {
  Args a = Args();
  a.parseArgs(args);
  std::shared_ptr<FastText> fasttext = std::make_shared<FastText>();
  std::string outputFileName;
  std::string outputFilename2;

  if (a.hasAutotune() &&
      a.getAutotuneModelSize() != Args::kUnlimitedModelSize) {
    outputFileName = a.output + ".ftz";
    outputFilename2 = a.output_word + ".ftz";
  } else {
    outputFileName = a.output + ".bin";
    outputFilename2 = a.output_word + ".bin";
  }
  std::ofstream ofs(outputFileName);
  std::ofstream ofs2(outputFilename2);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        outputFileName + " cannot be opened for saving.");
  }
  ofs.close();
  if (!ofs2.is_open()) {
      throw std::invalid_argument(
              outputFilename2 + "cannot be opened for saving.");
  }
  if (a.hasAutotune()) {
    Autotune autotune(fasttext);
    autotune.train(a);
  } else {
    fasttext->train(a);
  }
  printf("=========================\n");
  printf("=========================\n");
  fasttext->saveModel(outputFileName);
  printf("Model Saved \n");
  printf("=========================\n");
  printf("=========================\n");
  fasttext->save2Vectors(a.output + ".vec", a.output_word + ".vec");
  printf("Vector Saved\n");
  printf("=========================\n");
  printf("=========================\n");
  if (a.saveOutput) {
      fasttext->saveOutput(a.output + ".output");
      printf("Output Saved \n");
      printf("=========================\n");
      printf("=========================\n");
  }
}

void dump(const std::vector<std::string>& args) {
  if (args.size() < 4) {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }

  std::string modelPath = args[2];
  std::string option = args[3];

  FastText fasttext;
  fasttext.loadModel(modelPath);
  if (option == "args") {
    fasttext.getArgs().dump(std::cout);
  } else if (option == "dict") {
    fasttext.getDictionary()->dump(std::cout);
  } else if (option == "input") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getInputMatrix()->dump(std::cout);
    }
  } else if (option == "output") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getOutputMatrix()->dump(std::cout);
    }
  } else {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(args);
  } else if (command == "test" || command == "test-label") {
    test(args);
  } else if (command == "quantize") {
    quantize(args);
  } else if (command == "print-word-vectors") {
    printWordVectors(args);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(args);
  } else if (command == "print-ngrams") {
    printNgrams(args);
  } else if (command == "nn") {
      nn(args);
  } else if (command == "mrr") {
      mrr(args);
  } else if (command == "analogies") {
    analogies(args);
  } else if (command == "predict" || command == "predict-prob") {
    predict(args);
  } else if (command == "dump") {
    dump(args);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
