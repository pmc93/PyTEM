#include "PlotWidget.h"

#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QFontMetrics>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double niceScaleLength(double targetMetres)
{
    if (!(targetMetres > 0.0) || !std::isfinite(targetMetres))
        return 0.0;
    const double exponent = std::pow(10.0, std::floor(std::log10(targetMetres)));
    const double normalized = targetMetres / exponent;
    const double leading = normalized >= 5.0 ? 5.0
                         : normalized >= 2.0 ? 2.0 : 1.0;
    return leading * exponent;
}

} // namespace

PlotWidget::PlotWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(200, 180);
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

void PlotWidget::setBackgroundImage(const QImage &image, const QRectF &dataBounds,
                                    QString attribution)
{
    m_backgroundImage = image;
    m_backgroundBounds = dataBounds.normalized();
    m_backgroundAttribution = std::move(attribution);
    update();
}

void PlotWidget::clearBackgroundImage()
{
    m_backgroundImage = {};
    m_backgroundBounds = {};
    m_backgroundAttribution.clear();
    update();
}

void PlotWidget::setEqualAspect(bool enabled, double xScale)
{
    m_equalAspect = enabled;
    m_xAspectScale = std::max(1.0e-12, xScale);
    update();
}

void PlotWidget::setGeographicScaleBar(bool enabled)
{
    m_geographicScaleBar = enabled;
    update();
}

void PlotWidget::setLegendBackground(bool enabled)
{
    m_legendBackground = enabled;
    update();
}

void PlotWidget::setTickLabelsVisible(bool visible)
{
    m_tickLabelsVisible = visible;
    update();
}

void PlotWidget::setMinimumY(double value)
{
    m_minimumY = value;
    update();
}

void PlotWidget::setScientificX(bool enabled)
{
    m_scientificX = enabled;
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
    m_hitPoints.clear();
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor(250, 251, 252));

    const QRectF area(74, 38, std::max(40, width() - 100), std::max(40, height() - 105));
    painter.setPen(QColor(45, 52, 60));
    painter.drawText(QRectF(0, 7, width(), 25), Qt::AlignCenter, m_title);
    painter.drawRect(area);

    QVector<QPointF> valid;
    for (const auto &curve : m_curves) {
        for (int pointIndex = 0; pointIndex < curve.points.size(); ++pointIndex) {
            const auto &point = curve.points[pointIndex];
            if (std::isfinite(point.x()) && std::isfinite(point.y()) &&
                (!m_logX || point.x() > 0.0) && (!m_logY || point.y() > 0.0)) {
                valid.push_back(point);
                if (pointIndex < curve.yErrors.size()) {
                    const double error = std::abs(curve.yErrors[pointIndex]);
                    double lower = point.y() - error;
                    const double upper = point.y() + error;
                    if (m_logY && lower <= 0.0)
                        lower = point.y() * 0.1;
                    if (std::isfinite(lower) && (!m_logY || lower > 0.0))
                        valid.push_back({point.x(), lower});
                    if (std::isfinite(upper) && (!m_logY || upper > 0.0))
                        valid.push_back({point.x(), upper});
                }
            }
        }
    }
    // A background map is an underlay, not an axis-range constraint.  Using
    // the entire tile mosaic as data bounds forces tall map panels to expand
    // beyond the image and leaves large white bands.  Curves define the view;
    // the oversized tile mosaic is clipped to that view.
    if (valid.isEmpty() && !m_backgroundImage.isNull()
        && m_backgroundBounds.isValid()) {
        valid.push_back({m_backgroundBounds.left(), m_backgroundBounds.top()});
        valid.push_back({m_backgroundBounds.right(), m_backgroundBounds.bottom()});
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
    if (std::isfinite(m_minimumY) && (!m_logY || m_minimumY > 0.0))
        yMin = transformY(m_minimumY);

    if (m_equalAspect && !m_logX && !m_logY) {
        const double areaAspect = area.width() / area.height();
        double xSpan = xMax - xMin;
        double ySpan = yMax - yMin;
        if (xSpan * m_xAspectScale / ySpan < areaAspect) {
            const double expanded = ySpan * areaAspect / m_xAspectScale;
            const double center = 0.5 * (xMin + xMax);
            xMin = center - 0.5 * expanded;
            xMax = center + 0.5 * expanded;
        } else {
            const double expanded = xSpan * m_xAspectScale / areaAspect;
            const double center = 0.5 * (yMin + yMax);
            yMin = center - 0.5 * expanded;
            yMax = center + 0.5 * expanded;
        }
    }

    if (!m_backgroundImage.isNull() && m_backgroundBounds.isValid()) {
        const QPointF northWest = mapPoint(
            {m_backgroundBounds.left(), m_backgroundBounds.bottom()},
            area, xMin, xMax, yMin, yMax);
        const QPointF southEast = mapPoint(
            {m_backgroundBounds.right(), m_backgroundBounds.top()},
            area, xMin, xMax, yMin, yMax);
        painter.save();
        painter.setClipRect(area);
        painter.setOpacity(0.88);
        painter.drawImage(QRectF(northWest, southEast).normalized(),
                          m_backgroundImage);
        painter.restore();
    }

    painter.setFont(QFont(painter.font().family(), 8));
    for (int i = 0; i <= 5; ++i) {
        const double f = i / 5.0;
        const double px = area.left() + f * area.width();
        const double py = area.top() + f * area.height();
        painter.setPen(QColor(225, 228, 232));
        painter.drawLine(QPointF(px, area.top()), QPointF(px, area.bottom()));
        painter.drawLine(QPointF(area.left(), py), QPointF(area.right(), py));

        if (m_tickLabelsVisible) {
            painter.setPen(QColor(70, 75, 80));
            const double xv = xMin + f * (xMax - xMin);
            double yf = m_invertY ? f : 1.0 - f;
            const double yv = yMin + yf * (yMax - yMin);
            const double xValue = m_logX ? std::pow(10.0, xv) : xv;
            const QString xs = QString::number(xValue,
                                               m_scientificX ? 'e' : 'g',
                                               m_scientificX ? 2 : 3);
            const QString ys = QString::number(
                m_logY ? std::pow(10.0, yv) : yv, 'g', 3);
            painter.drawText(QRectF(px - 36, area.bottom() + 5, 72, 18),
                             Qt::AlignHCenter, xs);
            painter.drawText(QRectF(2, py - 9, 66, 18),
                             Qt::AlignRight | Qt::AlignVCenter, ys);
        }
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
        for (int pointIndex = 0; pointIndex < curve.points.size(); ++pointIndex) {
            const auto &point = curve.points[pointIndex];
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
            if (pointIndex < curve.yErrors.size()) {
                const double error = std::abs(curve.yErrors[pointIndex]);
                double lower = point.y() - error;
                const double upper = point.y() + error;
                if (m_logY && lower <= 0.0)
                    lower = point.y() * 0.1;
                if (error > 0.0 && std::isfinite(lower) && std::isfinite(upper)
                    && (!m_logY || (lower > 0.0 && upper > 0.0))) {
                    const QPointF lowPoint = mapPoint({point.x(), lower}, area,
                                                      xMin, xMax, yMin, yMax);
                    const QPointF highPoint = mapPoint({point.x(), upper}, area,
                                                       xMin, xMax, yMin, yMax);
                    painter.setPen(QPen(curve.color, 1.0));
                    painter.drawLine(lowPoint, highPoint);
                    painter.drawLine(QPointF(lowPoint.x() - 3.0, lowPoint.y()),
                                     QPointF(lowPoint.x() + 3.0, lowPoint.y()));
                    painter.drawLine(QPointF(highPoint.x() - 3.0, highPoint.y()),
                                     QPointF(highPoint.x() + 3.0, highPoint.y()));
                    painter.setPen(QPen(curve.color, 2.0));
                }
            }
            if (curve.markers)
                painter.drawEllipse(mapped, 3.2, 3.2);
            if (curve.selectable) {
                const int pointId = pointIndex < curve.pointIds.size()
                    ? curve.pointIds[pointIndex] : pointIndex;
                if (pointId >= 0)
                    m_hitPoints.push_back({mapped, pointId});
            }
        }
        if (curve.line)
            painter.drawPath(path);
    }

    if (m_geographicScaleBar && !m_logX && !m_logY) {
        constexpr double pi = 3.14159265358979323846;
        constexpr double metresPerLongitudeDegreeAtEquator = 111320.0;
        const double latitude = std::clamp(0.5 * (yMin + yMax), -89.0, 89.0);
        const double longitudeMetres = metresPerLongitudeDegreeAtEquator
            * std::max(0.01, std::abs(std::cos(latitude * pi / 180.0)));
        const double visibleWidthMetres = (xMax - xMin) * longitudeMetres;
        const double scaleMetres = niceScaleLength(0.22 * visibleWidthMetres);
        const double scalePixels = scaleMetres / visibleWidthMetres * area.width();
        if (scaleMetres > 0.0 && scalePixels >= 20.0) {
            const double left = area.left() + 14.0;
            const double baseline = area.bottom() - 14.0;
            const QString label = scaleMetres >= 1000.0
                ? QString("%1 km").arg(scaleMetres / 1000.0, 0, 'g', 3)
                : QString("%1 m").arg(scaleMetres, 0, 'g', 3);
            const QRectF backing(left - 8.0, baseline - 25.0,
                                 scalePixels + 16.0, 34.0);
            painter.save();
            painter.setPen(QPen(QColor(255, 255, 255, 225), 5.0));
            painter.drawLine(QPointF(left, baseline),
                             QPointF(left + scalePixels, baseline));
            painter.drawLine(QPointF(left, baseline - 5.0),
                             QPointF(left, baseline + 5.0));
            painter.drawLine(QPointF(left + scalePixels, baseline - 5.0),
                             QPointF(left + scalePixels, baseline + 5.0));
            painter.setPen(QPen(QColor(30, 30, 30), 2.0));
            painter.drawLine(QPointF(left, baseline),
                             QPointF(left + scalePixels, baseline));
            painter.drawLine(QPointF(left, baseline - 5.0),
                             QPointF(left, baseline + 5.0));
            painter.drawLine(QPointF(left + scalePixels, baseline - 5.0),
                             QPointF(left + scalePixels, baseline + 5.0));
            painter.setPen(QColor(30, 30, 30));
            painter.setBrush(QColor(255, 255, 255, 205));
            const QFontMetrics metrics(painter.font());
            const double labelWidth = metrics.horizontalAdvance(label) + 8.0;
            const QRectF labelRect(left + 0.5 * (scalePixels - labelWidth),
                                   backing.top(), labelWidth, 17.0);
            painter.drawRect(labelRect);
            painter.drawText(labelRect, Qt::AlignCenter, label);
            painter.restore();
        }
    }

    if (!m_backgroundAttribution.isEmpty()) {
        painter.setPen(QColor(35, 35, 35));
        painter.setBrush(QColor(255, 255, 255, 205));
        const QFontMetrics metrics(painter.font());
        const int attributionWidth = metrics.horizontalAdvance(m_backgroundAttribution) + 10;
        const QRectF attributionRect(area.right() - attributionWidth - 3,
                                     area.bottom() - 19,
                                     attributionWidth, 17);
        painter.drawRect(attributionRect);
        painter.drawText(attributionRect.adjusted(5, 0, -5, 0),
                         Qt::AlignCenter, m_backgroundAttribution);
        painter.setBrush(Qt::NoBrush);
    }

    int namedCurveCount = 0;
    int widestLegendText = 0;
    const QFontMetrics legendMetrics(painter.font());
    for (const auto &curve : m_curves) {
        if (curve.name.isEmpty())
            continue;
        ++namedCurveCount;
        widestLegendText = std::max(
            widestLegendText, legendMetrics.horizontalAdvance(curve.name));
    }
    const int legendContentWidth = std::max(125, 28 + widestLegendText);
    int legendX = static_cast<int>(area.right()) - legendContentWidth - 10;
    int legendY = static_cast<int>(area.top()) + 10;
    if (m_legendBackground && namedCurveCount > 0) {
        const QRectF legendBox(legendX - 8, legendY - 6,
                               legendContentWidth + 16,
                               namedCurveCount * 18 + 12);
        painter.setPen(QColor(165, 170, 175));
        painter.setBrush(QColor(255, 255, 255));
        painter.drawRect(legendBox);
        painter.setBrush(Qt::NoBrush);
    }
    for (const auto &curve : m_curves) {
        if (curve.name.isEmpty())
            continue;
        painter.setPen(QPen(curve.color, 2.0));
        if (curve.line)
            painter.drawLine(legendX, legendY + 6, legendX + 22, legendY + 6);
        else
            painter.drawEllipse(QPointF(legendX + 11, legendY + 6), 3.2, 3.2);
        painter.setPen(QColor(45, 52, 60));
        painter.drawText(legendX + 28, legendY + 11, curve.name);
        legendY += 18;
    }
}

void PlotWidget::mousePressEvent(QMouseEvent *event)
{
    if ((event->button() != Qt::LeftButton && event->button() != Qt::RightButton)
        || m_hitPoints.isEmpty()) {
        QWidget::mousePressEvent(event);
        return;
    }
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    const QPointF click = event->position();
#else
    const QPointF click = event->localPos();
#endif
    double bestDistanceSquared = 100.0;
    int bestId = -1;
    for (const auto &hit : m_hitPoints) {
        const double dx = click.x() - hit.position.x();
        const double dy = click.y() - hit.position.y();
        const double distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestId = hit.pointId;
        }
    }
    if (bestId >= 0) {
        if (event->button() == Qt::RightButton)
            emit pointRightClicked(bestId);
        else
            emit pointClicked(bestId);
    }
    else
        QWidget::mousePressEvent(event);
}
