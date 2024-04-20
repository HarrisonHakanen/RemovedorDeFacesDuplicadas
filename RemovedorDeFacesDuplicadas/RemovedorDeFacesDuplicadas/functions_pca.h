#ifndef PCA_H
#define PCA_H

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

struct EigenSort {
    std::vector<float> valores;
    std::vector<std::vector<float>> vetores;
};

std::vector<std::vector<float>> pcaFit(std::vector<std::vector<float>> matrizDeDescritores,bool ordenacao);
cv::Mat extraiMatrizConvarianca(std::vector<std::vector<float>> matriz, int linha);
std::vector<std::vector<float>> criaMatrizIdentidade(int coluna);
cv::Mat transporMatriz(std::vector<std::vector<float>> matriz);
std::vector<std::vector<float>> reduzDimensao(int numComponentes, std::vector<std::vector<float>> vetoresOrdenados);
EigenSort ordenarAutoVetoresValores(cv::Mat eigenValues, cv::Mat eigenVectors);
std::vector<std::vector<float>> multiplicacaoDeMatrizes(cv::Mat matrizDescritoresTransposta, cv::Mat vetoresOrdenadosTransposta);
void escreverArquivoPca(std::string nomeArquivo, auto conteudo);
void sobreescreverArquivo(std::string nomeArquivo, auto conteudo);
std::vector<float> extraiMediaDeMatriz(std::vector<std::vector<float>> matriz, int linha,int coluna);
float round_to(float value, double precision);
void salvaAutoValores(cv::Mat eigenValues);
void salvarAutoVetores(cv::Mat eigenVectors);
std::vector<std::vector<float>> subtraiMatrizPorMedias(std::vector<std::vector<float>> matrizDeDescritores, std::vector<float>mediaColunas, std::vector<float>desviosPadroes);
float arredondaNumero(float numero, int casasDecimais);
cv::Mat vectorToMat(std::vector<std::vector<float>> vetor);
std::vector<std::vector<float>> matToVector(cv::Mat matriz);
std::vector<float> extraiDesvioPadrao(std::vector<float>mediaColunas, std::vector<std::vector<float>> matrizDeDescritores);
#endif