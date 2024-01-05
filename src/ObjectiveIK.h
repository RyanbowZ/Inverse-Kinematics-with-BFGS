#pragma once

#ifndef ObjectiveIK_H
#define ObjectiveIK_H

#include "Objective.h"

class ObjectiveIK : public Objective
{
public:
    ObjectiveIK(double a = 1.0, double b = 1.0);
    void setWeights(Eigen::Vector3d wt){wtar=wt(0),wreg=wt(2);wroot=wt(1);}
    void setTarget(Eigen::Vector2d pt){ptar=pt;}
    virtual ~ObjectiveIK();
    virtual double evalObjective(const Eigen::VectorXd &x) const;
    virtual double evalObjective(const Eigen::VectorXd &x, Eigen::VectorXd &g) const;
    virtual double evalObjective(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H) const;

private:
    double a;
    double b;
    double wtar=1000;
    double wreg=1;
    double wroot=0;
    Eigen::Vector2d ptar;
    Eigen::Vector3d r;
    Eigen::Matrix3d T0;
    Eigen::Matrix3d T;
};

#endif
