#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/serialize.h>
#include <dlib/matrix.h>
#include <dlib/opencv/cv_image.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <filesystem>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <stdio.h>


using namespace std;
namespace fs = std::filesystem;
using namespace cv;
using namespace dlib;


template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

struct Pessoa {
    string pastaDaPessoa;
    std::vector<Pessoa>pessoasIguais;
    std::vector<float> mediaDosDescritores;
};



double normDifference(const std::vector<float>vec1, const std::vector<float>vec2);
std::vector<float> retornaMedia(std::vector<std::vector<float>>vetores);

int main()
{
    frontal_face_detector detector_face = get_frontal_face_detector();

    shape_predictor detector_pontos;
    deserialize("modelos_treinados\\shape_predictor_68_face_landmarks.dat") >> detector_pontos;

    CascadeClassifier haarcascade;
    haarcascade.load("C:\\Users\\harri\\Documents\\GitHub\\ColetorDeFaces\\ColetorDeFaces\\ColetorDeFaces\\modelos_treinados\\haarcascade_frontalface_alt.xml");

    anet_type extrator_descritor_facial;
    deserialize("modelos_treinados\\dlib_face_recognition_resnet_model_v1.dat") >> extrator_descritor_facial;



    string root = "C:\\Users\\harri\\Documents\\Programacao\\Python\\CelebV-HQ-main\\CelebV-HQ-main\\downloaded_celebvhq_final";


    for (const auto& pastaPrincipal : fs::directory_iterator(root)) {

        std::vector<string>subPastas;

        for (const auto& pastasImagens : fs::directory_iterator(pastaPrincipal)) {
            
            if (fs::is_directory(pastasImagens.path().string())) {                
                subPastas.push_back(pastasImagens.path().string());                
            }
        }        
        if (subPastas.size() > 1) {
            
            std::vector<Pessoa>listaDePessoas;

            for (int i = 0; i < subPastas.size(); i++) {

                try
                {
                    cout << "Analisando a pasta: " << subPastas[i] << "\n";

                    std::vector<std::vector<float>> descritoresAindaNaoAtribuidos;
                    int contadorFrames = 0;

                    std::vector<matrix<rgb_pixel>> faces;

                    for (const auto& caminhoImagem : fs::directory_iterator(subPastas[i])) {

                        bool coletouFeatures = false;
                        if (contadorFrames < 5) {

                            Mat grayscale_image;
                            std::vector<Rect> features;

                            Mat imagem = imread(caminhoImagem.path().string());

                            cvtColor(imagem, grayscale_image, COLOR_BGR2GRAY);
                            equalizeHist(grayscale_image, grayscale_image);

                            haarcascade.detectMultiScale(grayscale_image, features, 1.1, 4, 0, Size(30, 30));

                            for (auto&& feature : features) {

                                coletouFeatures = true;
                                array2d<bgr_pixel> dlibImg;
                                assign_image(dlibImg, cv_image<bgr_pixel>(imagem));

                                for (auto face : detector_face(dlibImg)) {

                                    full_object_detection pontos = detector_pontos(dlibImg, face);
                                    matrix<rgb_pixel> face_chip;
                                    extract_image_chip(dlibImg, get_face_chip_details(pontos, 150, 0.25), face_chip);
                                    faces.push_back(std::move(face_chip));
                                }
                            }
                        }
                        else {

                            std::vector<matrix<float, 0, 1>> face_descriptors = extrator_descritor_facial(faces);

                            for (int indexDesc = 0; indexDesc < face_descriptors.size(); indexDesc++) {

                                std::vector<float> descritoresValue;

                                for (long j = 0; j < face_descriptors[indexDesc].nr(); ++j) {
                                    descritoresValue.push_back(face_descriptors[indexDesc](j));
                                }

                                descritoresAindaNaoAtribuidos.push_back(descritoresValue);
                            }


                            Pessoa pessoa = { subPastas[i],{},retornaMedia(descritoresAindaNaoAtribuidos) };

                            if (listaDePessoas.size() == 0) {

                                listaDePessoas.push_back(pessoa);
                            }
                            else {

                                std::vector<double>listaDeDiferencas;
                                bool pessoaJaExisteNaLista = false;

                                for (int indPessoa = 0; indPessoa < listaDePessoas.size(); indPessoa++) {

                                    double diferenca = normDifference(pessoa.mediaDosDescritores, listaDePessoas[indPessoa].mediaDosDescritores);

                                    if (diferenca < 0.6) {
                                        listaDePessoas[indPessoa].pessoasIguais.push_back(pessoa);
                                        pessoaJaExisteNaLista = true;
                                        indPessoa = listaDePessoas.size();
                                    }
                                }

                                if (!pessoaJaExisteNaLista) {
                                    listaDePessoas.push_back(pessoa);
                                }
                            }


                            break;
                        }
                        if (coletouFeatures) {
                            contadorFrames += 1;
                        }

                    }
                }
                catch (const std::exception&)
                {
                    cout << "Erro na pasta: " << subPastas[i] << "\n";
                }
            }

            for (int indP = 0; indP < listaDePessoas.size(); indP++) {

                std::vector<string>pastasDaPessoa;
                pastasDaPessoa.push_back(listaDePessoas[indP].pastaDaPessoa);

                std::vector<string>pastaSplit = split(listaDePessoas[indP].pastaDaPessoa, "\\");

                string nomeNovaPasta = "";
                for (int indSplit = 0; indSplit < pastaSplit.size() - 1; indSplit++) {
                    nomeNovaPasta += pastaSplit[indSplit] + "\\";
                }
                nomeNovaPasta += "Pessoa" + to_string(indP);

                if (!fs::is_directory(nomeNovaPasta)) {
                    fs::create_directory(nomeNovaPasta);
                }



                for (int indPessoasIguais = 0; indPessoasIguais < listaDePessoas[indP].pessoasIguais.size(); indPessoasIguais++) {
                    pastasDaPessoa.push_back(listaDePessoas[indP].pessoasIguais[indPessoasIguais].pastaDaPessoa);
                }

                for (int indPasta = 0; indPasta < pastasDaPessoa.size(); indPasta++) {

                    for (const auto& imagem : fs::directory_iterator(pastasDaPessoa[indPasta])) {

                        std::vector<string> imagemPathSplit = split(imagem.path().string(), "\\");

                        string nomeArquivo = imagemPathSplit[imagemPathSplit.size() - 1];
                        string pastaArquivo = imagemPathSplit[imagemPathSplit.size() - 2];

                        string novoArquivo = pastaArquivo + nomeArquivo;

                        string novoCaminho = nomeNovaPasta + "\\" + novoArquivo;

                        fs::copy(imagem.path().string(), novoCaminho);
                    }
                    fs::remove_all(pastasDaPessoa[indPasta]);
                }
            }        
        }
    }
}



double normDifference(const std::vector<float>vec1, const std::vector<float>vec2) {
    
    if (vec1.size() != vec2.size()) {
        std::cerr << "Erro: Os vetores têm tamanhos diferentes." << std::endl;
        return -1;
    }

    double sumOfSquares = 0.0;

    
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = vec1[i] - vec2[i];
        sumOfSquares += diff * diff;
    }
    
    return std::sqrt(sumOfSquares);
}

std::vector<float> retornaMedia(std::vector<std::vector<float>>vetores) {

    std::vector<float> medias;


    for (int linha = 0; linha < vetores[0].size(); linha++) {

        float somatoria = 0;
        for (int coluna = 0; coluna < vetores.size(); coluna++) {

            somatoria += vetores[coluna][linha];
        }

        medias.push_back(somatoria / vetores.size());
    }
    return medias;
}