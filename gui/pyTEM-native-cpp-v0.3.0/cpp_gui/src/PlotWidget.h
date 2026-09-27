#pragma once

#include <QColor>
#include <QPointF>
#include <QString>
#include <QVector>
#include <QWidget>

class PlotWidget final : public QWidget
{
    Q_OBJECT

public:
    struct Curve {
        QString name;
        QVector<QPointF> points;
        QColor color;
        bool markers = false;
        bool stepped = false;
    };

    explicit PlotWidget(QWidget *parent = nullptr);

    void setAxes(QString title, QString xLabel, QString yLabel,
                 bool logX, bool logY, bool invertY = false);
    void setCurves(QVector<Curve> curves);
    void clear();

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QPointF mapPoint(const QPointF &point, const QRectF &area,
                     double xMin, double xMax, double yMin, double yMax) const;
    double transformX(double value) const;
    double transformY(double value) const;

    QString m_title;
    QString m_xLabel;
    QString m_yLabel;
    QVector<Curve> m_curves;
    bool m_logX = true;
    bool m_logY = true;
    bool m_invertY = false;
};

