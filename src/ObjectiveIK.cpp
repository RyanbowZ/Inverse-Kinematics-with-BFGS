#include "ObjectiveIK.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace Eigen;
using namespace std;

ObjectiveIK::ObjectiveIK(double a, double b)
{
    this->a = a;
    this->b = b;
    this->r<< 1.0,
        0.0,
        1.0;
    this->T<<1.0,0.0,1.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0;
    this->T0<<1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0;
}

ObjectiveIK::~ObjectiveIK()
{
    
}

double ObjectiveIK::evalObjective(const VectorXd &x) const
{
    int n=(int)x.size();
    Matrix3d P_;
    P_.setIdentity();
    for(int i = 0; i < n; ++i){
        double a=x(i);
        Matrix3d Ri0;
        Ri0<<cos(a), -sin(a), 0.0,
        sin(a),cos(a),0.0,
        0.0,0.0,1.0;
        if(i){
            P_=P_*T;
        }
        else{
            P_=P_*T0;
        }
        P_=P_*Ri0;
    }
    Vector2d p0=(P_*r).segment<2>(0);
    Vector2d dp=p0-ptar;
    MatrixXd Wreg(n,n);
    Wreg.setIdentity();
    if(n>2)Wreg(0,0)=wroot;
    VectorXd ta=x.transpose();
    double f1=wtar*dp.transpose()*dp,f2=ta.transpose()*Wreg*ta;
    double f=0.5*(f1+f2);
    return f;
}

double ObjectiveIK::evalObjective(const VectorXd &x, VectorXd &g) const
{
    int n=(int)x.size();
    
    MatrixXd Wreg(n,n);
    Wreg.setIdentity();
    if(n>2)Wreg(0,0)=wroot;
    
    Matrix3d P_;
    P_.setIdentity();
    vector<Matrix3d> R0;
    vector<Matrix3d> R1;
    vector<Matrix3d> R2;
    for(int i = 0; i < n; ++i){
        double a=x(i);
        Matrix3d Ri0;
        Ri0<<cos(a), -sin(a), 0.0,
        sin(a),cos(a),0.0,
        0.0,0.0,1.0;
        Matrix3d Ri1;
        Ri1<<-sin(a), -cos(a), 0.0,
        cos(a),-sin(a),0.0,
        0.0,0.0,0.0;
        Matrix3d Ri2;
        Ri2<<-cos(a), sin(a), 0.0,
        -sin(a),-cos(a),0.0,
        0.0,0.0,0.0;
        R0.emplace_back(Ri0);
        R1.emplace_back(Ri1);
        R2.emplace_back(Ri2);
        
        if(i){
            P_=P_*T;
        }
        else{
            P_=P_*T0;
        }
        P_=P_*Ri0;
    }
    Vector2d p0=(P_*r).segment<2>(0);
    Vector2d dp=p0-ptar;
    
    VectorXd ta=x.transpose();
    double f1=wtar*dp.transpose()*dp,f2=ta.transpose()*Wreg*ta;
    double f=0.5*(f1+f2);
    
    MatrixXd p1(2,n);
    
    for(int j = 0; j < n; ++j){
        Matrix3d P_1;
        P_1.setIdentity();
        for(int i = 0; i < n; ++i){
            if(i){
                P_1=P_1*T;
            }
            else{
                P_1=P_1*T0;
            }
            if(i==j){
                P_1=P_1*R1[i];
            }
            else{
                P_1=P_1*R0[i];
            }
        }
        p1.block<2,1>(0,j)=(P_1*r).segment<2>(0);
    }
    
    g=wtar*(dp.transpose()*p1).transpose()+Wreg*x;
    
    return f;
}

double ObjectiveIK::evalObjective(const VectorXd &x, VectorXd &g, MatrixXd &H) const
{
    int n=(int)x.size();
    
    MatrixXd Wreg(n,n);
    Wreg.setIdentity();
    if(n>2)Wreg(0,0)=wroot;
    
    Matrix3d P_;
    P_.setIdentity();
    vector<Matrix3d> R0;
    vector<Matrix3d> R1;
    vector<Matrix3d> R2;
    for(int i = 0; i < n; ++i){
        double a=x(i);
        Matrix3d Ri0;
        Ri0<<cos(a), -sin(a), 0.0,
        sin(a),cos(a),0.0,
        0.0,0.0,1.0;
        Matrix3d Ri1;
        Ri1<<-sin(a), -cos(a), 0.0,
        cos(a),-sin(a),0.0,
        0.0,0.0,0.0;
        Matrix3d Ri2;
        Ri2<<-cos(a), sin(a), 0.0,
        -sin(a),-cos(a),0.0,
        0.0,0.0,0.0;
        R0.emplace_back(Ri0);
        R1.emplace_back(Ri1);
        R2.emplace_back(Ri2);
        
        if(i){
            P_=P_*T;
        }
        else{
            P_=P_*T0;
        }
        P_=P_*Ri0;
    }
    Vector2d p0=(P_*r).segment<2>(0);
    Vector2d dp=p0-ptar;
    
    VectorXd ta=x.transpose();
    double f1=wtar*dp.transpose()*dp,f2=ta.transpose()*Wreg*ta;
    double f=0.5*(f1+f2);
    
    MatrixXd p1(2,n);
    
    for(int j = 0; j < n; ++j){
        Matrix3d P_1;
        P_1.setIdentity();
        for(int i = 0; i < n; ++i){
            if(i){
                P_1=P_1*T;
            }
            else{
                P_1=P_1*T0;
            }
            if(i==j){
                P_1=P_1*R1[i];
            }
            else{
                P_1=P_1*R0[i];
            }
        }
        p1.block<2,1>(0,j)=(P_1*r).segment<2>(0);
    }
    
    g=wtar*(dp.transpose()*p1).transpose()+Wreg*x;
    
    MatrixXd p2(2*n,n);
    
    for(int j = 0; j < n; ++j){
        for(int k = 0; k < n; ++k){
            Matrix3d P_2;
            P_2.setIdentity();
            for(int i = 0; i < n; ++i){
                if(i){
                    P_2=P_2*T;
                }
                else{
                    P_2=P_2*T0;
                }
                if(i==j&&j==k){
                    P_2=P_2*R2[i];
                }
                else if(i==j||i==k){
                    P_2=P_2*R1[i];
                }
                else{
                    P_2=P_2*R0[i];
                }
            }
            p2.block<2,1>(2*k,j)=(P_2*r).segment<2>(0);
        }
    }
    //cout<<p2<<endl;
    MatrixXd p2H(n,n);
    for(int j = 0; j < n; ++j){
        for(int k = 0; k < n; ++k){
            Vector2d p2jk=p2.block<2,1>(2*k,j);
            p2H(k,j)=dp.transpose()*p2jk;
        }
    }
    H=wtar*(p1.transpose()*p1+p2H).transpose()+Wreg;
    
    
    return f;
//    double a=x(0);
//    std::cout<<"theta:= "<<a<<std::endl;
//    Matrix3d Ri0;
//    Ri0<<cos(a), -sin(a), 0.0,
//    sin(a),cos(a),0.0,
//    0.0,0.0,1.0;
//    Matrix3d Ri1;
//    Ri1<<-sin(a), -cos(a), 0.0,
//    cos(a),-sin(a),0.0,
//    0.0,0.0,0.0;
//    Matrix3d Ri2;
//    Ri2<<-cos(a), sin(a), 0.0,
//    -sin(a),-cos(a),0.0,
//    0.0,0.0,0.0;
//    Vector2d p0;
//    p0=(T*Ri0*r).segment<2>(0);
//    Vector2d p1;
//    p1=(T*Ri1*r).segment<2>(0);
//    Vector2d p2;
//    p2=(T*Ri2*r).segment<2>(0);
//    Vector2d dp;
//    dp=p0-ptar;
//    
//    double f=0.5*(wtar*dp.transpose()*dp+wreg*pow(a,2));
//    g<<wtar*dp.transpose()*p1+wreg*a;
//    double w1=wtar*p1.transpose()*p1;
//    double w2=wtar*dp.transpose()*p2;
//    //double w1=wtar*(p1.transpose()*p1+dp.transpose()*p2);
//    H<<w1+w2+wreg;
//    return f;
}
