#include "PlotWidget.h"

#include <QPainter>
#include <QPainterPath>
#include <algorithm>
#include <cmath>
#include <limits>

PlotWidget::PlotWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(440, 280);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void PlotWidget::setAxes(QString title, QString xLabel, QString yLabel,
                         bool logX, bool logY, bool invertY)
{
    m_title = std::move(title);
    m_xLabel = std::move(xLabel);
    m_yLabel = std::move(yLabel);
    m_logX = logX;
    m_logY = logY;
    m_invertY = invertY;
    update();
}

void PlotWidget::setCurves(QVector<Curve> curves)
{
    m_curves = std::move(curves);
    update();
}

void PlotWidget::clear()
{
    m_curves.clear();
    update();
}

double PlotWidget::transformX(double value) const
{
    return m_logX ? std::log10(value) : value;
}

double PlotWidget::transformY(double value) const
{
    return m_logY ? std::log10(value) : value;
}

QPointF PlotWidget::mapPoint(const QPointF &point, const QRectF &area,
                             double xMin, double xMax, double yMin, double yMax) const
{
    const double x = transformX(point.x());
    const double y = transformY(point.y());
    const double px = area.left() + (x - xMin) / (xMax - xMin) * area.width();
    double fraction = (y - yMin) / (yMax - yMin);
    if (!m_invertY)
        fraction = 1.0 - fraction;
    const double py = area.top() + fraction * area.height();
    return {px, py};
}

void PlotWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor(250, 251, 252));

    const QRectF area(74, 38, std::max(40, width() - 100), std::max(40, height() - 105));
    painter.setPen(QColor(45, 52, 60));
    painter.drawText(QRectF(0, 7, width(), 25), Qt::AlignCenter, m_title);
    painter.drawRect(area);

    QVector<QPointF> valid;
    for (const auto &curve : m_curves) {
        for (const auto &point : curve.points) {
            if (std::isfinite(point.x()) && std::isfinite(point.y()) &&
                (!m_logX || point.x() > 0.0) && (!m_logY || point.y() > 0.0))
                valid.push_back(point);
        }
    }

    if (valid.isEmpty()) {
        painter.setPen(QColor(120, 125, 130));
        painter.drawText(area, Qt::AlignCenter, "Load data and run an inversion");
        return;
    }

    double xMin = std::numeric_limits<double>::max();
    double xMax = std::numeric_limits<double>::lowest();
    double yMin = std::numeric_limits<double>::max();
    double yMax = std::numeric_limits<double>::lowest();
    for (const auto &point : valid) {
        xMin = std::min(xMin, transformX(point.x()));
        xMax = std::max(xMax, transformX(point.x()));
        yMin = std::min(yMin, transformY(point.y()));
        yMax = std::max(yMax, transformY(point.y()));
    }
    if (xMax <= xMin) { xMin -= 0.5; xMax += 0.5; }
    if (yMax <= yMin) { yMin -= 0.5; yMax += 0.5; }
    const double xPad = 0.06 * (xMax - xMin);
    const double yPad = 0.08 * (yMax - yMin);
    xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

    painter.setFont(QFont(painter.font().family(), 8));
    for (int i = 0; i <= 5; ++i) {
        const double f = i / 5.0;
        const double px = area.left() + f * area.width();
        const double py = area.top() + f * area.height();
        painter.setPen(QColor(225, 228, 232));
        painter.drawLine(QPointF(px, area.top()), QPointF(px, area.bottom()));
        painter.drawLine(QPointF(area.left(), py), QPointF(area.right(), py));

        painter.setPen(QColor(70, 75, 80));
        const double xv = xMin + f * (xMax - xMin);
        double yf = m_invertY ? f : 1.0 - f;
        const double yv = yMin + yf * (yMax - yMin);
        const QString xs = QString::number(m_logX ? std::pow(10.0, xv) : xv, 'g', 3);
        const QString ys = QString::number(m_logY ? std::pow(10.0, yv) : yv, 'g', 3);
        painter.drawText(QRectF(px - 36, area.bottom() + 5, 72, 18), Qt::AlignHCenter, xs);
        painter.drawText(QRectF(2, py - 9, 66, 18), Qt::AlignRight | Qt::AlignVCenter, ys);
    }

    painter.setFont(QFont(painter.font().family(), 9));
    painter.drawText(QRectF(area.left(), height() - 30, area.width(), 20), Qt::AlignCenter, m_xLabel);
    painter.save();
    painter.translate(18, area.center().y());
    painter.rotate(-90);
    painter.drawText(QRectF(-area.height() / 2, -10, area.height(), 20), Qt::AlignCenter, m_yLabel);
    painter.restore();

    for (const auto &curve : m_curves) {
        painter.setPen(QPen(curve.color, 2.0));
        QPainterPath path;
        bool started = false;
        QPointF previous;
        for (const auto &point : curve.points) {
            if (!std::isfinite(point.x()) || !std::isfinite(point.y()) ||
                (m_logX && point.x() <= 0.0) || (m_logY && point.y() <= 0.0))
                continue;
            const QPointF mapped = mapPoint(point, area, xMin, xMax, yMin, yMax);
            if (!started) {
                path.moveTo(mapped);
                started = true;
            } else if (curve.stepped) {
                path.lineTo(mapped.x(), previous.y());
                path.lineTo(mapped);
            } else {
                path.lineTo(mapped);
            }
            previous = mapped;
            if (curve.markers)
                painter.drawEllipse(mapped, 3.2, 3.2);
        }
        painter.drawPath(path);
    }

    int legendX = static_cast<int>(area.right()) - 135;
    int legendY = static_cast<int>(area.top()) + 10;
    for (const auto &curve : m_curves) {
        painter.setPen(QPen(curve.color, 2.0));
        painter.drawLine(legendX, legendY + 6, legendX + 22, legendY + 6);
        painter.setPen(QColor(45, 52, 60));
        painter.drawText(legendX + 28, legendY + 11, curve.name);
        legendY += 18;
    }
}

