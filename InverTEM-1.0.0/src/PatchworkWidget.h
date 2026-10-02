// Patchwork QC view (as in EEMstudio): a sounding-by-gate image per moment,
// coloured by the data, data / response, the misfit per gate or the relative
// STD, with the gates used and the RMS of each sounding below. Grey cells are
// gates not used; a dot marks negative data.
#pragma once

#include <QString>
#include <QVector>
#include <QWidget>

#include <functional>

class PatchworkWidget final : public QWidget
{
    Q_OBJECT

public:
    struct Cell { double value = 0.0; bool valid = false, used = false, negative = false; };
    struct Panel { QString title; int gates = 0; QVector<QVector<Cell>> columns; }; // columns[sounding][gate]
    struct Scale { QString title; double minimum = 0.0, maximum = 1.0; bool logarithmic = false; bool diverging = false; };

    explicit PatchworkWidget(QWidget *parent = nullptr);
    void setData(QVector<Panel> panels, QVector<double> distances, QVector<int> gatesUsed, QVector<double> rms,
                 Scale scale, int activeColumn);

signals:
    void columnClicked(int column);
    void cellRightClicked(int panel, int column, int gate);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

private:
    QColor colour(double value) const;
    bool cellAt(const QPointF &point, int &panel, int &column, int &gate) const;

    QVector<Panel> m_panels;
    QVector<double> m_distances, m_rms;
    QVector<int> m_gatesUsed;
    Scale m_scale;
    int m_active = -1;
    QVector<QRectF> m_panelRects; // filled when painted
};
