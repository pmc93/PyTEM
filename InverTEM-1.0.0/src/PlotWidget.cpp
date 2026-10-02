#include "PlotWidget.h"

#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <QFontMetrics>
#include <QLinearGradient>
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

QColor interpolateColor(const QColor &first, const QColor &second,
                        double fraction)
{
    fraction = std::clamp(fraction, 0.0, 1.0);
    return QColor::fromRgbF(
        first.redF() + fraction * (second.redF() - first.redF()),
        first.greenF() + fraction * (second.greenF() - first.greenF()),
        first.blueF() + fraction * (second.blueF() - first.blueF()));
}

} // namespace

PlotWidget::PlotWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(200, 120);
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

void PlotWidget::setLegendCorner(Qt::Corner corner)
{
    m_legendCorner = corner;
    update();
}

void PlotWidget::setLegendToggle(bool enabled)
{
    m_legendToggle = enabled;
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

void PlotWidget::setMarkerRadius(double radius)
{
    m_markerRadius = std::clamp(radius, 1.5, 12.0);
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

void PlotWidget::setYTickIntervals(int intervals)
{
    m_yTickIntervals = std::max(1, intervals);
    update();
}

void PlotWidget::setNiceLogXTicks(bool enabled)
{
    m_niceLogXTicks = enabled;
    update();
}

void PlotWidget::setXRange(double minimum, double maximum)
{
    if (std::isfinite(minimum) && std::isfinite(maximum)
        && minimum < maximum && (!m_logX || minimum > 0.0)) {
        m_xMinimum = minimum;
        m_xMaximum = maximum;
    }
    update();
}

void PlotWidget::clearXRange()
{
    m_xMinimum = std::numeric_limits<double>::quiet_NaN();
    m_xMaximum = std::numeric_limits<double>::quiet_NaN();
    update();
}

void PlotWidget::setColorScale(QString title, double minimum, double maximum)
{
    m_colorScaleTitle = std::move(title);
    m_colorScaleMinimum = minimum;
    m_colorScaleMaximum = maximum;
    update();
}

void PlotWidget::clearColorScale()
{
    m_colorScaleTitle.clear();
    m_colorScaleMinimum = std::numeric_limits<double>::quiet_NaN();
    m_colorScaleMaximum = std::numeric_limits<double>::quiet_NaN();
    update();
}

QColor PlotWidget::colorScaleColor(double value, double minimum,
                                  double maximum)
{
    if (!(maximum > minimum) || !std::isfinite(value))
        return QColor("#808080");
    const double fraction = std::clamp(
        (value - minimum) / (maximum - minimum), 0.0, 1.0);
    // Low RMS is good (green), intermediate RMS is yellow, and high RMS is
    // poor (red): the requested ColorBrewer-style GnYlRd direction.
    const QColor low("#1a9850");
    const QColor middle("#fee08b");
    const QColor high("#d73027");
    return fraction <= 0.5
        ? interpolateColor(low, middle, fraction * 2.0)
        : interpolateColor(middle, high, (fraction - 0.5) * 2.0);
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
    painter.setPen(Qt::black);
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
    // A single map sounding has no natural coordinate span.  Use a compact
    // geographic viewport instead of the generic +/-0.5 fallback (roughly
    // 100 km), which made both the point and its tile background appear
    // missing. Other plots retain the original fallback.
    const double degenerateHalfSpan = m_geographicScaleBar ? 0.001 : 0.5;
    if (xMax <= xMin) {
        xMin -= degenerateHalfSpan;
        xMax += degenerateHalfSpan;
    }
    if (yMax <= yMin) {
        yMin -= degenerateHalfSpan;
        yMax += degenerateHalfSpan;
    }
    const double xPad = 0.06 * (xMax - xMin);
    const double yPad = 0.08 * (yMax - yMin);
    xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
    if (std::isfinite(m_xMinimum) && std::isfinite(m_xMaximum)
        && m_xMaximum > m_xMinimum
        && (!m_logX || m_xMinimum > 0.0)) {
        xMin = transformX(m_xMinimum);
        xMax = transformX(m_xMaximum);
    }
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
    // Horizontal grid and y-axis labels.
    for (int i = 0; i <= m_yTickIntervals; ++i) {
        const double f = i / static_cast<double>(m_yTickIntervals);
        const double py = area.top() + f * area.height();
        painter.setPen(QColor(225, 228, 232));
        painter.drawLine(QPointF(area.left(), py), QPointF(area.right(), py));

        if (m_tickLabelsVisible) {
            painter.setPen(Qt::black);
            double yf = m_invertY ? f : 1.0 - f;
            const double yv = yMin + yf * (yMax - yMin);
            const QString ys = QString::number(
                m_logY ? std::pow(10.0, yv) : yv, 'g', 3);
            painter.drawText(QRectF(2, py - 9, 66, 18),
                             Qt::AlignRight | Qt::AlignVCenter, ys);
        }
    }

    // Vertical grid and x-axis labels. Resistivity plots use log-friendly
    // 1/2/5 ticks rather than arbitrary equal pixel divisions.
    if (m_logX && m_niceLogXTicks) {
        const double visibleDecades = xMax - xMin;
        const int firstExponent = static_cast<int>(std::floor(xMin)) - 1;
        const int lastExponent = static_cast<int>(std::ceil(xMax)) + 1;
        for (int exponent = firstExponent; exponent <= lastExponent; ++exponent) {
            const double decade = std::pow(10.0, exponent);
            for (double multiplier : {1.0, 2.0, 5.0}) {
                const double value = multiplier * decade;
                const double transformed = std::log10(value);
                if (transformed < xMin - 1.0e-12
                    || transformed > xMax + 1.0e-12)
                    continue;
                const double px = area.left()
                    + (transformed - xMin) / (xMax - xMin) * area.width();
                painter.setPen(QColor(225, 228, 232));
                painter.drawLine(QPointF(px, area.top()),
                                 QPointF(px, area.bottom()));
                if (m_tickLabelsVisible
                    && (visibleDecades <= 2.2 || multiplier == 1.0)) {
                    painter.setPen(Qt::black);
                    painter.drawText(
                        QRectF(px - 38, area.bottom() + 5, 76, 18),
                        Qt::AlignHCenter, QString::number(value, 'g', 3));
                }
            }
        }
    } else {
        const int divisions = m_scientificX ? 2 : 5; // three ticks on the time axis
        for (int i = 0; i <= divisions; ++i) {
            const double fraction = i / static_cast<double>(divisions);
            const double px = area.left() + fraction * area.width();
            painter.setPen(QColor(225, 228, 232));
            painter.drawLine(QPointF(px, area.top()),
                             QPointF(px, area.bottom()));
            if (m_tickLabelsVisible) {
                painter.setPen(Qt::black);
                const double transformed = xMin + fraction * (xMax - xMin);
                const double value = m_logX ? std::pow(10.0, transformed)
                                             : transformed;
                const QString label = QString::number(
                    value, m_scientificX ? 'e' : 'g',
                    m_scientificX ? 2 : 3);
                painter.drawText(QRectF(px - 36, area.bottom() + 5, 72, 18),
                                 Qt::AlignHCenter, label);
            }
        }
    }

    painter.setFont(QFont(painter.font().family(), 9));
    painter.setPen(Qt::black);
    painter.drawText(QRectF(area.left(), height() - 30, area.width(), 20), Qt::AlignCenter, m_xLabel);
    painter.save();
    painter.translate(18, area.center().y());
    painter.rotate(-90);
    painter.drawText(QRectF(-area.height() / 2, -10, area.height(), 20), Qt::AlignCenter, m_yLabel);
    painter.restore();

    for (const auto &curve : m_curves) {
        painter.setPen(QPen(curve.color, 2.0,
                            curve.dashed ? Qt::DashLine : Qt::SolidLine));
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
                    painter.setPen(QPen(curve.color, 2.0,
                                        curve.dashed ? Qt::DashLine
                                                     : Qt::SolidLine));
                }
            }
            if (curve.markers) {
                if (curve.filledMarkers)
                    painter.setBrush(curve.color);
                else
                    painter.setBrush(Qt::NoBrush);
                painter.drawEllipse(mapped, m_markerRadius, m_markerRadius);
                painter.setBrush(Qt::NoBrush);
            }
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
            painter.setPen(QColor(145, 150, 155));
            painter.setBrush(QColor(255, 255, 255));
            painter.drawRect(backing);
            painter.setBrush(Qt::NoBrush);
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
            const QFontMetrics metrics(painter.font());
            const double labelWidth = metrics.horizontalAdvance(label) + 8.0;
            const QRectF labelRect(left + 0.5 * (scalePixels - labelWidth),
                                   backing.top(), labelWidth, 17.0);
            painter.drawText(labelRect, Qt::AlignCenter, label);
            painter.restore();
        }
    }

    if (!m_colorScaleTitle.isEmpty()
        && std::isfinite(m_colorScaleMinimum)
        && std::isfinite(m_colorScaleMaximum)
        && m_colorScaleMaximum > m_colorScaleMinimum) {
        const double barHeight = std::min(130.0, area.height() * 0.32);
        const double barWidth = 13.0;
        const double barX = area.right() - barWidth - 12.0;
        const double barY = area.bottom() - barHeight - 12.0;
        QLinearGradient gradient(0.0, barY, 0.0, barY + barHeight);
        gradient.setColorAt(0.0, colorScaleColor(
            m_colorScaleMaximum, m_colorScaleMinimum, m_colorScaleMaximum));
        gradient.setColorAt(0.5, colorScaleColor(
            0.5 * (m_colorScaleMinimum + m_colorScaleMaximum),
            m_colorScaleMinimum, m_colorScaleMaximum));
        gradient.setColorAt(1.0, colorScaleColor(
            m_colorScaleMinimum, m_colorScaleMinimum, m_colorScaleMaximum));
        painter.save();
        // As wide as the title needs ("Elevation (m)"), right-aligned.
        const double boxWidth = std::max(barWidth + 59.0,
            QFontMetrics(painter.font()).horizontalAdvance(m_colorScaleTitle) + 10.0);
        const QRectF scaleBox(barX + barWidth + 7.0 - boxWidth, barY - 25.0,
                              boxWidth, barHeight + 32.0);
        painter.setPen(QColor(145, 150, 155));
        painter.setBrush(QColor(255, 255, 255));
        painter.drawRect(scaleBox);
        painter.setPen(Qt::black);
        painter.setBrush(gradient);
        painter.drawRect(QRectF(barX, barY, barWidth, barHeight));
        painter.setBrush(Qt::NoBrush);
        painter.drawText(QRectF(scaleBox.left(), barY - 20.0, boxWidth, 18.0),
                         Qt::AlignCenter, m_colorScaleTitle);
        painter.setBrush(Qt::NoBrush);
        for (int tick = 0; tick <= 5; ++tick) {
            const double fraction = tick / 5.0;
            const double y = barY + (1.0 - fraction) * barHeight;
            const double value = m_colorScaleMinimum
                + fraction * (m_colorScaleMaximum - m_colorScaleMinimum);
            painter.drawLine(QPointF(barX - 4.0, y), QPointF(barX, y));
            painter.drawText(QRectF(barX - 45.0, y - 8.0, 38.0, 16.0),
                             Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(value, 'f', 2));
        }
        painter.restore();
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
    const bool left = m_legendCorner == Qt::TopLeftCorner || m_legendCorner == Qt::BottomLeftCorner;
    const bool bottom = m_legendCorner == Qt::BottomLeftCorner || m_legendCorner == Qt::BottomRightCorner;
    const bool hidden = m_legendToggle && m_legendHidden;
    const int boxWidth = hidden ? legendMetrics.horizontalAdvance("Legend") + 16 : legendContentWidth + 16;
    const int boxHeight = hidden ? 20 : namedCurveCount * 18 + 12;
    m_legendRect = QRectF(left ? area.left() + 2 : area.right() - boxWidth - 2,
                          bottom ? area.bottom() - boxHeight - 2 : area.top() + 4, boxWidth, boxHeight)
                       .translated(m_legendOffset);
    const QPointF unclamped = m_legendRect.topLeft();
    m_legendRect.moveTo(std::clamp(m_legendRect.left(), 0.0, std::max(0.0, width() - m_legendRect.width())),
                        std::clamp(m_legendRect.top(), 0.0, std::max(0.0, height() - m_legendRect.height())));
    m_legendOffset += m_legendRect.topLeft() - unclamped; // kept inside the widget
    if (namedCurveCount == 0)
        m_legendRect = QRectF();
    if (namedCurveCount > 0 && (m_legendBackground || hidden)) {
        painter.setPen(QColor(165, 170, 175));
        painter.setBrush(QColor(255, 255, 255));
        painter.drawRect(m_legendRect);
        painter.setBrush(Qt::NoBrush);
    }
    if (namedCurveCount > 0 && hidden) {
        painter.setPen(QColor(45, 52, 60));
        painter.drawText(m_legendRect, Qt::AlignCenter, "Legend");
    }
    int legendX = static_cast<int>(m_legendRect.left()) + 8;
    int legendY = static_cast<int>(m_legendRect.top()) + 6;
    for (const auto &curve : m_curves) {
        if (curve.name.isEmpty() || hidden)
            continue;
        painter.setPen(QPen(curve.color, 2.0,
                            curve.dashed ? Qt::DashLine : Qt::SolidLine));
        if (curve.line)
            painter.drawLine(legendX, legendY + 6, legendX + 22, legendY + 6);
        else {
            if (curve.filledMarkers)
                painter.setBrush(curve.color);
            else
                painter.setBrush(Qt::NoBrush);
            painter.drawEllipse(QPointF(legendX + 11, legendY + 6), 3.2, 3.2);
            painter.setBrush(Qt::NoBrush);
        }
        painter.setPen(QColor(45, 52, 60));
        painter.drawText(legendX + 28, legendY + 11, curve.name);
        legendY += 18;
    }

    if (m_rightDragging) {
        painter.save();
        painter.setPen(QPen(QColor("#2b73b6"), 1.5, Qt::DashLine));
        painter.setBrush(QColor(43, 115, 182, 35));
        painter.drawRect(QRectF(m_dragStart, m_dragCurrent).normalized());
        painter.restore();
    }
}

void PlotWidget::mousePressEvent(QMouseEvent *event)
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    const QPointF position = event->position();
#else
    const QPointF position = event->localPos();
#endif
    if (event->button() == Qt::LeftButton && m_legendRect.contains(position)) {
        m_legendDragging = true;
        m_legendMoved = false;
        m_legendPress = position;
        m_legendPressOffset = m_legendOffset;
        return;
    }
    if ((event->button() != Qt::LeftButton && event->button() != Qt::RightButton)
        || m_hitPoints.isEmpty()) {
        QWidget::mousePressEvent(event);
        return;
    }
    const QPointF &click = position;
    if (event->button() == Qt::RightButton) {
        m_rightDragging = true;
        m_dragStart = click;
        m_dragCurrent = click;
        event->accept();
        update();
        return;
    }
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
        emit pointClicked(bestId);
    }
    else
        QWidget::mousePressEvent(event);
}

void PlotWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_legendDragging) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        const QPointF shift = event->position() - m_legendPress;
#else
        const QPointF shift = event->localPos() - m_legendPress;
#endif
        m_legendMoved = m_legendMoved || shift.manhattanLength() > 3.0;
        if (m_legendMoved) {
            m_legendOffset = m_legendPressOffset + shift;
            update();
        }
        return;
    }
    if (!m_rightDragging) {
        QWidget::mouseMoveEvent(event);
        return;
    }
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    m_dragCurrent = event->position();
#else
    m_dragCurrent = event->localPos();
#endif
    event->accept();
    update();
}

void PlotWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (m_legendDragging && event->button() == Qt::LeftButton) {
        m_legendDragging = false;
        if (!m_legendMoved && m_legendToggle) { // a click, not a drag
            m_legendHidden = !m_legendHidden;
            update();
        }
        return;
    }
    if (!m_rightDragging || event->button() != Qt::RightButton) {
        QWidget::mouseReleaseEvent(event);
        return;
    }
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    m_dragCurrent = event->position();
#else
    m_dragCurrent = event->localPos();
#endif
    m_rightDragging = false;
    const QRectF selection(m_dragStart, m_dragCurrent);
    const QRectF normalized = selection.normalized();
    constexpr double dragThreshold = 5.0;
    if (event->modifiers().testFlag(Qt::ControlModifier)) { // Ctrl: select, without editing
        QVector<int> selected;
        for (const auto &hit : m_hitPoints)
            if (normalized.adjusted(-6, -6, 6, 6).contains(hit.position) && !selected.contains(hit.pointId))
                selected.push_back(hit.pointId);
        emit pointsSelected(selected);
    } else if (normalized.width() < dragThreshold
        && normalized.height() < dragThreshold) {
        double bestDistanceSquared = 100.0;
        int bestId = -1;
        for (const auto &hit : m_hitPoints) {
            const double dx = m_dragCurrent.x() - hit.position.x();
            const double dy = m_dragCurrent.y() - hit.position.y();
            const double distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < bestDistanceSquared) {
                bestDistanceSquared = distanceSquared;
                bestId = hit.pointId;
            }
        }
        if (bestId >= 0)
            emit pointRightClicked(bestId);
    } else {
        QVector<int> selected;
        for (const auto &hit : m_hitPoints) {
            if (normalized.contains(hit.position)
                && !selected.contains(hit.pointId))
                selected.push_back(hit.pointId);
        }
        if (!selected.isEmpty())
            emit pointsRightDragged(selected, event->modifiers().testFlag(Qt::ShiftModifier));
    }
    event->accept();
    update();
}
