#include "OptimizerBFGS.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerBFGS::OptimizerBFGS()
{
	
}

OptimizerBFGS::~OptimizerBFGS()
{
	
}

VectorXd OptimizerBFGS::optimize(const shared_ptr<Objective> objective, const VectorXd &xInit)
{
    int n=(int)xInit.size();
    
    MatrixXd A(n,n);
    A.setIdentity();
    MatrixXd I(n,n);
    I.setIdentity();
    VectorXd x0(n);
    VectorXd g0(n);
    VectorXd x=xInit;
    VectorXd ans;
    VectorXd dx;
    
    
    for(iter=0;iter<iterMax;iter++){
        
        VectorXd g(n);
        double f=objective->evalObjective(x,g);
        if(iter>0){
            VectorXd s=x-x0;
            VectorXd y=g-g0;
            double r=y.transpose()*s;
            double raw=1/r;
            A=(I-raw*s*y.transpose())*A*(I-raw*y*s.transpose())+raw*s*s.transpose();
        }
        VectorXd p = -A*g;
        double alpha=alphaInit;
        for(int iterLS=0;iterLS<iterMaxLS;iterLS++){
            dx=alpha*p;
            double f1=objective->evalObjective(x+dx);
            if(f1<f)break;
            alpha*=gamma;
        }
        x0=x;
        g0=g;
        x+=dx;
        if(g.norm()<tol)break;
    }
    
	return x;
}
