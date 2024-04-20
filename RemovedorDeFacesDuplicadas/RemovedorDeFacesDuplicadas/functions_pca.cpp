#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <filesystem>

#include "functions_pca.h"
#include "global_pca.h"




std::vector<std::vector<float>> pcaFit(std::vector<std::vector<float>> matrizDeDescritores,bool ordenacao) {


    std::vector<std::vector<float>> matrizResultante;        

    //Extrair e salva medias dos valores  
    std::vector<float>mediaColunas = extraiMediaDeMatriz(matrizDeDescritores, matrizDeDescritores.size(), numDimensions);

    std::string mediasContent = "";
    for (int i = 0; i < mediaColunas.size(); i++) {
        mediasContent += std::to_string(mediaColunas[i]) + "\n";
    }
    //sobreescreverArquivo(mediasFile, mediasContent);

    //Extrair e salva desvio padrão
    std::vector<float>desviosPadroes = extraiDesvioPadrao(mediaColunas, matrizDeDescritores);

    std::string desviosContent = "";
    for (int i = 0; i < desviosPadroes.size(); i++) {
        desviosContent += std::to_string(desviosPadroes[i]) + "\n";
    }
    //sobreescreverArquivo(desviosPadroesFile, desviosContent);

    matrizDeDescritores = subtraiMatrizPorMedias(matrizDeDescritores, mediaColunas, desviosPadroes);

    //Extrai a covariança da matriz.
    cv::Mat matrizDeCovarianca = extraiMatrizConvarianca(matrizDeDescritores, matrizDeDescritores.size()); 


    //Extrai os autovalores e os autovetores
    cv::Mat eigenValues, eigenVectors;
    cv::eigen(matrizDeCovarianca, eigenValues, eigenVectors);    
    
    eigenVectors = eigenVectors.t();

    //Salva os autovalores e os autovetores           
    salvaAutoValores(eigenValues);
    salvarAutoVetores(eigenVectors);

    EigenSort eigenSort = {};

    if (ordenacao) {        
        eigenSort = ordenarAutoVetoresValores(eigenValues, eigenVectors);
    }
    else {
        std::vector<std::vector<float>> vetoresOrdenados = matToVector(eigenVectors);
        eigenSort.vetores = vetoresOrdenados;
    }
    
    //Remove quantidade de colunas do EigenVectors com base na quantidade de colunas desejada
    std::vector<std::vector<float>> vetoresReduzidos = reduzDimensao(numComponents, eigenSort.vetores);


    cv::Mat matDescritores = vectorToMat(matrizDeDescritores);
    cv::Mat matReduzidos = vectorToMat(vetoresReduzidos);

    //Calcula o produto escalar de duas matrizes que já estão transpostas                                                                              
    matrizResultante = multiplicacaoDeMatrizes(matDescritores, matReduzidos);


    return matrizResultante;
}

std::vector<std::vector<float>> matToVector(cv::Mat matriz) {

    std::vector<std::vector<float>> matrizVec;

    for (int i = 0; i < matriz.rows; i++) {

        std::vector<float> valores;
        for (int j = 0; j < matriz.cols; j++) {

            valores.push_back(matriz.at<float>(i, j));            
        }
        matrizVec.push_back(valores);
    }
    return matrizVec;
}

cv::Mat vectorToMat(std::vector<std::vector<float>> vetor) {

    cv::Mat mat(vetor.size(), vetor[0].size(), CV_32S);

    for (int i = 0; i < vetor.size(); i++) {
        for (int j = 0; j < vetor[i].size(); j++) {

            mat.at<float>(i, j) = vetor[i][j];
        }
    }
    return mat;
}


std::vector<std::vector<float>> subtraiMatrizPorMedias(std::vector<std::vector<float>> matrizDeDescritores, std::vector<float>mediaColunas, std::vector<float>desviosPadroes) {

    for (int i = 0; i < matrizDeDescritores.size(); i++) {
        for (int j = 0; j < matrizDeDescritores[0].size(); j++) {
            matrizDeDescritores[i][j] = (matrizDeDescritores[i][j] - mediaColunas[j]) / desviosPadroes[j];            
        }
    }
    return matrizDeDescritores;
}


void salvaAutoValores(cv::Mat eigenValues) {

    std::string conteudo = "";

    for (int i = 0; i < eigenValues.rows; i++) {

        for (int j = 0; j < eigenValues.cols; j++) {

            std::string value = std::to_string(eigenValues.at<float>(i, j));
            if (value != "") {

                if (i + 1 < eigenValues.rows) {
                    conteudo += value + "\n";
                }
                else {
                    conteudo += value;
                }
            }
        }
    }
    //sobreescreverArquivo(eigenValuesFile, conteudo);
}


void salvarAutoVetores(cv::Mat eigenVectors) {

    std::string conteudo = "";

    if (eigenVectors.cols > 128) {
        std::cout << "teste";
    }


    for (int i = 0; i < eigenVectors.rows; i++) {

        for (int j = 0; j < eigenVectors.cols; j++) {

            std::string value = std::to_string(eigenVectors.at<float>(i, j));

            if (j + 1 < eigenVectors.cols) {
                conteudo += value + "{";
            }
            else {
                conteudo += value;
            }

        }
        if (i + 1 < eigenVectors.rows) {
            conteudo += "\n";
        }
    }
    //sobreescreverArquivo(eigenVectorsFile, conteudo);
}


std::vector<std::vector<float>> criaMatrizIdentidade(int coluna) {

    std::vector<std::vector<float>> matrizIdentidade(coluna, std::vector<float>(coluna));;

    for (int i = 0; i < coluna; i++) {
        for (int j = 0; j < coluna; j++) {

            if (i == j) {
                matrizIdentidade[i][j] = 1;
            }
            else {
                matrizIdentidade[i][j] = 0;
            }
        }
    }

    return matrizIdentidade;
}

std::vector<float> extraiMediaDeMatriz(std::vector<std::vector<float>> matriz, int linha, int coluna) {

    std::vector<float>mediaColunas;

    if (linha > 0) {

        if (matriz[0].size() >= coluna) {

            for (int i = 0; i < coluna; i++) {

                float valor = 0;
                float valorAtual = 0;
                for (int j = 0; j < linha; j++) {

                    valorAtual = matriz[j][i];
                    valor += valorAtual;
                }
                mediaColunas.push_back(valor / linha);
            }
        }
        else {
            std::cout << "Menor que coluna";
        }
    }
    else {
        std::cout << "Menor que linha";
    }



    return mediaColunas;
}


std::vector<float> extraiDesvioPadrao(std::vector<float>mediaColunas, std::vector<std::vector<float>> matrizDeDescritores) {

    std::vector<float>desviosPadroes;
    for (int i = 0; i < matrizDeDescritores[0].size(); i++) {
        float somatoria = 0;
        for (int j = 0; j < matrizDeDescritores.size(); j++) {
            somatoria += std::pow(matrizDeDescritores[j][i] - mediaColunas[i], 2);
        }        

        float div = somatoria / (matrizDeDescritores.size()-1);
        float raiz = std::sqrt(div);        
        desviosPadroes.push_back(raiz);
    }

    return desviosPadroes;
}


cv::Mat extraiMatrizConvarianca(std::vector<std::vector<float>> matrizMedia, int linha) {


    cv::Mat matrizDeCovarianca(numDimensions, numDimensions, CV_32F);

    for (int comparador = 0; comparador < numDimensions; comparador++) {

        for (int comparada = 0; comparada < numDimensions; comparada++) {

            float soma = 0;
            for (int j = 0; j < linha; j++) {

                soma += matrizMedia[j][comparador] * matrizMedia[j][comparada];
            }
            matrizDeCovarianca.at<float>(comparador, comparada) = soma / (linha-1);

        }
    }

    return matrizDeCovarianca;
}

cv::Mat transporMatriz(std::vector<std::vector<float>> matriz) {

    cv::Mat matrizTransposta(matriz[0].size(), matriz.size(), CV_32F);

    for (int i = 0; i < matriz[0].size(); i++) {

        for (int j = 0; j < matriz.size(); j++) {

            matrizTransposta.at<float>(i, j) = matriz[j][i];
        }
    }

    return matrizTransposta;
}

std::vector<std::vector<float>> reduzDimensao(int numComponentes, std::vector<std::vector<float>> vetoresOrdenados) {

    std::vector<std::vector<float>> vetoresReduzidos;


    for (int i = 0; i < vetoresOrdenados.size(); i++) {

        std::vector<float> valoresReduzidos;
        for (int j = 0; j < numComponentes; j++) {

            valoresReduzidos.push_back(vetoresOrdenados[i][j]);
        }
        vetoresReduzidos.push_back(valoresReduzidos);
    }

    return vetoresReduzidos;
}

EigenSort ordenarAutoVetoresValores(cv::Mat eigenValues, cv::Mat eigenVectors) {

    std::vector<float> valoresOrdenados;
    for (int i = eigenValues.rows - 1; i > -1; i--) {

        for (int j = eigenValues.cols - 1; j > -1; j--) {
            valoresOrdenados.push_back(eigenValues.at<float>(i, j));
        }
    }

    std::vector<std::vector<float>> vetoresOrdenados;
    for (int i = 0; i < eigenVectors.rows; ++i) {

        std::vector<float>vetor;
        for (int j = 0; j < eigenVectors.cols; ++j) {
            vetor.push_back(eigenVectors.at<float>(i, j));
        }
        std::vector<float>vetorOrdenado;
        for (int j = vetor.size() - 1; j > -1; j--) {
            vetorOrdenado.push_back(vetor[j]);
        }
        vetoresOrdenados.push_back(vetorOrdenado);
    }


    EigenSort eigenSort = { valoresOrdenados,vetoresOrdenados };
    return eigenSort;
}


std::vector<std::vector<float>> multiplicacaoDeMatrizes(cv::Mat matrizDescritoresTransposta, cv::Mat vetoresOrdenadosTransposta) {

    cv::Size dimensoesMatrizDescritores = matrizDescritoresTransposta.size();
    cv::Size dimensoesVetoresOrdenados = vetoresOrdenadosTransposta.size();

    int rows_A = dimensoesMatrizDescritores.height;
    int cols_A = dimensoesMatrizDescritores.width;
    int cols_B = dimensoesVetoresOrdenados.width;

    std::vector<std::vector<float>> matrizResultante(rows_A, std::vector<float>(cols_B, 0));

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            for (int k = 0; k < cols_A; k++) {
                matrizResultante[i][j] += matrizDescritoresTransposta.at<float>(i, k) * vetoresOrdenadosTransposta.at<float>(k, j);
            }
        }
    }

    return matrizResultante;
}


void sobreescreverArquivo(std::string nomeArquivo, auto conteudo) {

    std::ofstream arquivoCadastro(nomeArquivo);
    arquivoCadastro << conteudo;
    arquivoCadastro.close();
}


void escreverArquivoPca(std::string nomeArquivo, auto conteudo) {
    
    std::ofstream arquivoCadastro(nomeArquivo);
    arquivoCadastro << conteudo;
    arquivoCadastro.close();
}

float round_to(float value, double precision) {
    return std::round(value / precision) * precision;
}

float arredondaNumero(float numero, int casasDecimais) {
    float fator = std::pow(10, casasDecimais);
    return std::round(numero * fator) / fator;
}
