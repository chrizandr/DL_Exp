#include "tiny_dnn/tiny_dnn.h" // order matters, this has to go before opencv (ACCESS_WRITE)
#include <opencv2/opencv.hpp>
#include <iostream>
#include <glob.h>

using namespace cv;
using namespace std;
using tiny_dnn::vec_t;
using tiny_dnn::label_t;


int WINSIZE   = 24; // dnn/json will override this
String tscdir = "datasets/";


//! convert a single cv::Mat image plane to tiny_dnn::vec_t
void convert_plane(const Mat &img, vec_t& tiny_vec)
{
    const float scale = 1.0f / 255.0f;
    Mat_<uchar> image = img.reshape(1, 1);
    std::transform(image.begin(), image.end(), std::back_inserter(tiny_vec),
                   [=](uint8_t c) { return float(c) * scale; });
}


//! convert (possibly color) image to tiny_dnn::vec_t, with consecutive image planes
void convert_image(const Mat &image, vec_t & tiny_vec)
{
    if (image.channels() == 1) {
        convert_plane(image, tiny_vec);
    } else {
        vector<Mat> chn;
        split(image, chn);
        for (auto c : chn) {
            convert_plane(c, tiny_vec);
        }
    }
}


//! convert Mat to vec_t for tiny_dnn, and add to dataset
//  tiny_dnn wants consecutive image planes
void add_image(const Mat &image, int lab, std::vector<vec_t>& data, std::vector<label_t>& labels)
{
    vec_t tiny_vec;
    convert_image(image, tiny_vec);
    data.push_back(tiny_vec);
    labels.push_back(lab);
    cout << "dnn " << data.size() << "\t" << lab << "\r";
}


//! convert image Mat for cv::ml classes, and add to dataset
void add_image(const Mat &image, int lab, cv::Mat &data, cv::Mat &labels)
{
    Mat img;
    image.convertTo(img, CV_32F, 1.0/255);
    data.push_back(img.reshape(1,1));
    labels.push_back(lab);
    cout << "mat " << data.rows << "\t" << lab << "\r";
}


//! load images for max_classes classes from train or test dir.
//  note, that the csv files *claim* to have W,h,x,y order, but actually, it's the other way round ! (H,W,y,x)
template<class Datatype, class Labelstype>
double load(const String &dir, Datatype &data, Labelstype &labels, int max_classes=-1, bool gray=true, int skip=0)
{
    // int64 t0 = getTickCount();
    int k = 0;
    cout << dir << endl;
    vector<String> csvs;
    glob(dir + "*.csv", csvs, true);
    for (auto cn : csvs) {
        ifstream csv(cn);
        string file;
        getline(csv, file); // skip csv header
        while(csv.good()) {
            k++;
            char c = 0;
            file = "";
            // avoid "evil string globbing"
            while (csv.good()) {
                csv.get(c);
                if (c == ';') break;
                if (c == '\n') continue;
                file += c;
            }
            if (file.empty()) break;

            int W,H,x1,y1,x2,y2,label;
            csv >> H >> c >> W >> c >> y1 >> c >> x1 >> c >> y2 >> c >> x2 >> c >> label;
            if ((skip > 1) && (k % skip != 0)) continue;
            if ((max_classes > 0) && (label >= max_classes)) break;

            String fn = dir + format("%05d/",label) + file;
            Mat img = imread(fn, (gray?0:1));
            if (img.empty()) continue;

            cv::Mat resized;
            Rect roi(Point(x1,y1), Point(x2,y2));
            cv::resize(img(roi), resized, cv::Size(WINSIZE, WINSIZE));
            add_image(resized, label, data, labels);
        }
    }


    // int64 t1 = getTickCount();
    // return  ((t1-t0)/getTickFrequency());
}

//! load a json model from file, adjust traindata settings (winsize, max_classes)
//!  optionally load pretrained weights
template <class Optimizer>
int dnn_train(const string &json_model, const string &pre_weigths, float learn, float decay, int batch_size)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;
    typedef cross_entropy loss_t;

    network<sequential> nn;

    int max_classes = 62;
    try {
        nn.load(json_model.c_str(), content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        max_classes = shp_out[0].width_;

        if (! pre_weigths.empty()) {
            cout << "reading weights from " << pre_weigths << endl;
            ifstream ifs(pre_weigths.c_str());
            ifs >> nn;
        } else {
            //nn.weight_init(weight_init::xavier(1));
            //nn.weight_init(weight_init::lecun());
        }

        nn.save("mymodel.txt", content_type::model, file_format::json);
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    Optimizer opt;
    opt.alpha = learn;

    cout << "in " << WINSIZE << ", out " << max_classes << ", batch " << batch_size << endl;
    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer: " << i << "\n";
        cout << "type: "   << nn[i]->layer_type() << "\n";
        cout << "input: "  << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output: " << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

    // load data
    vector<vec_t>   t_data,   v_data;
    vector<label_t> t_labels, v_labels;

    double tl = load(tscdir + "Training/", t_data, t_labels, max_classes, false);
    int n = t_data.size() * t_data[0].size();
    cout << "dnn train " << t_data.size() << " samples, " << n << " bytes. " << tl << " seconds. " << endl;

    tl = load(tscdir + "Testing/", v_data, v_labels, max_classes, false, 3); //only 1/3 of test set
    n = v_data.size() * v_data[0].size();
    cout << "dnn test  " << v_data.size() << " samples, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    size_t z=0; // samples per epoch
    size_t epochs = 0;
    size_t count = 0; // overall samples in this training pass
    float best_result=0;

    auto on_enumerate_epoch = [&](){
        opt.alpha *= decay;  // continuously decay learning rate
        opt.alpha = std::max((tiny_dnn::float_t)0.00001, opt.alpha);
        std::cout << "epoch " << epochs << " " << count << " samples " << t.elapsed();
        std::cout << " seconds, " << opt.alpha << " alpha, ";

        result res = nn.test(v_data, v_labels);
        float accuracy = (float(res.num_success) / res.num_total);
        cout << accuracy << " accuracy." << endl;
        //double loss = nn.get_loss<loss_t>(v_data, v_labels);
        //cout << "loss " << loss << endl;

        // save weights
        if (accuracy > best_result) {
            std::ofstream ofs("my.net");
            ofs << nn;
            best_result = accuracy;
        }
        //
        // for (int i=0; i<nn.depth()-1; i++)
        //     nn[i]->output_to_image().write(format("layer%i.bmp", i));

        t.restart();
        z = 0; // reset local counter
        epochs ++;
    };

    auto on_enumerate_data = [&](){
        cout << (z+=batch_size) << "\r"; // local
        count += batch_size;             // global
    };

    nn.train<loss_t>(opt, t_data, t_labels, batch_size, 1000,
                  on_enumerate_data, on_enumerate_epoch);
    return 0;
}

//! load a json model and pretrained weights from file, adjust traindata settings (winsize, max_classes)
//!  and predict on test images
int dnn_test(const string &json_model, const string &pre_weigths)
{
    using namespace tiny_dnn;

    network<sequential> nn;
    int max_classes = 62;
    try {
        nn.load(json_model, content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        max_classes = shp_out[0].width_;

        cout << "reading weights from " << pre_weigths << endl;
        ifstream ifs(pre_weigths);
        ifs >> nn;
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    // load data
    vector<vec_t>   v_data;
    vector<label_t> v_labels;

    double tl = load(tscdir + "Testing/", v_data, v_labels, max_classes, false, 1);
    int n = v_data.size() * v_data[0].size();
    cout << "dnn test  " << v_data.size() << " samples, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    cout << "check1" << endl;

    for(int i=0; i<v_data[0].size(); i++){
        cout << v_data[0][i] << endl;
    }

    result res = nn.test(v_data, v_labels);
    cout << "check2" << endl;
    float accuracy = (float(res.num_success) / res.num_total);
    cout << "test " << accuracy << " accuracy, " << t.elapsed() << " seconds." << endl;
    return 0;
}


int main(int argc, char **argv)
{
    using namespace tiny_dnn;

    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ svm s          |      | reference svm test }"
        "{ mlp m          |      | reference mlp test }"
        "{ test t         |      | test dnn on pretrained model }"
        "{ batch b        |    24| batch size for dnn training }"
        "{ maxc M         |    62| restrict to M classes (for speedup), only for svm,mlp }"
        "{ learn l        |0.0004| initial learning rate for dnn }"
        "{ decay d        |0.95  | decay learning rate per epoch }"
        "{ weights w      |      | pretrained weights file (my.net) }"
        "{ optimizer o    |grad  | optimizer for dnn training }"
        "{ json j         |tsc32.txt| json model file for dnn (required) }"
        "{ data D         |/home/chrizandr/DL_Exp/CS231n/Traffic_Sign/datasets/| path to dataset }" );

    string json(parser.get<string>("json"));
    if (parser.has("help") || json.empty())
    {
        parser.printMessage();
        return 0;
    }
    tscdir = parser.get<string>("data"); // global
    string saved(parser.get<string>("weights"));
    string opt(parser.get<string>("optimizer"));
    int max_classes = parser.get<int>("maxc");
    int batch_size = parser.get<int>("batch");
    float learn = parser.get<float>("learn");
    float decay = parser.get<float>("decay");
    // bool do_svm = parser.has("svm");
    // bool do_mlp = parser.has("mlp");
    bool do_test = parser.has("test");

    // if (do_svm)
    //     return cv_svm(max_classes);
    // if (do_mlp)
    //     return cv_mlp(max_classes);
    if (do_test)
        return dnn_test(json, saved);

    // if (opt == "rms")
    //     return dnn_train<RMSprop>(json, saved, learn, decay, batch_size);
    // if (opt == "adam")
    //     return dnn_train<adam>(json, saved, learn, decay, batch_size);
    // if (opt == "adagrad")
    //     return dnn_train<adagrad>(json, saved, learn, decay, batch_size);
    // if (opt == "momentum")
    //     return dnn_train<momentum>(json, saved, learn, decay, batch_size);
    return dnn_train<gradient_descent>(json, saved, learn, decay, batch_size);
}
