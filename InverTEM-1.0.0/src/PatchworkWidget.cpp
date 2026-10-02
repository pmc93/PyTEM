#include "PatchworkWidget.h"

#include <QMouseEvent>
#include <QPainter>

#include <algorithm>
#include <cmath>

PatchworkWidget::PatchworkWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumSize(300, 240);
}

void PatchworkWidget::setData(QVector<Panel> panels, QVector<double> distances, QVector<int> gatesUsed, QVector<double> rms,
                              Scale scale, int activeColumn)
{
    m_panels = std::move(panels);
    m_distances = std::move(distances);
    m_gatesUsed = std::move(gatesUsed);
    m_rms = std::move(rms);
    m_scale = std::move(scale);
    m_active = activeColumn;
    update();
}

// Viridis-like ramp, or blue - white - red around 1 (ratios) or 0 (misfits).
QColor PatchworkWidget::colour(double value) const
{
    double f = m_scale.logarithmic
        ? (std::log10(value) - std::log10(m_scale.minimum)) / (std::log10(m_scale.maximum) - std::log10(m_scale.minimum))
        : (value - m_scale.minimum) / (m_scale.maximum - m_scale.minimum);
    f = std::clamp(std::isfinite(f) ? f : 0.0, 0.0, 1.0);
    static const QColor viridis[] = {"#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"};
    static const QColor diverging[] = {"#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"};
    const QColor *stops = m_scale.diverging ? diverging : viridis;
    const double position = f * 4.0;
    const int i = std::min(3, static_cast<int>(position));
    const double t = position - i;
    return QColor::fromRgbF(stops[i].redF() + t * (stops[i + 1].redF() - stops[i].redF()),
                            stops[i].greenF() + t * (stops[i + 1].greenF() - stops[i].greenF()),
                            stops[i].blueF() + t * (stops[i + 1].blueF() - stops[i].blueF()));
}

void PatchworkWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.fillRect(rect(), QColor(250, 251, 252));
    m_panelRects.clear();
    const int columns = static_cast<int>(m_distances.size());
    if (columns == 0 || m_panels.isEmpty()) {
        painter.setPen(QColor(120, 125, 130));
        painter.drawText(rect(), Qt::AlignCenter, "Load data to see the patchwork");
        return;
    }
    // Panels share the height above a 70 px strip in proportion to their gate counts.
    const double left = 74.0, right = width() - 96.0, top = 30.0, stripHeight = 70.0, gap = 34.0;
    const double bottom = height() - 40.0 - stripHeight - gap;
    int totalGates = 0;
    for (const Panel &panel : m_panels)
        totalGates += std::max(1, panel.gates);
    const double rowHeight = std::max(1.0, (bottom - top - gap * (m_panels.size() - 1)) / totalGates);
    const double columnWidth = (right - left) / columns;
    double y = top;
    painter.setFont(QFont(painter.font().family(), 8));
    for (const Panel &panel : m_panels) {
        const QRectF area(left, y, right - left, rowHeight * std::max(1, panel.gates));
        m_panelRects.push_back(area);
        painter.setPen(Qt::black);
        painter.drawText(QRectF(0, y - 22, width(), 20), Qt::AlignCenter, panel.title);
        for (int c = 0; c < std::min(columns, static_cast<int>(panel.columns.size())); ++c)
            for (int g = 0; g < panel.columns[c].size(); ++g) {
                const Cell &cell = panel.columns[c][g];
                const QRectF box(left + c * columnWidth, y + g * rowHeight, columnWidth + 0.5, rowHeight + 0.5);
                if (!cell.valid) {
                    painter.fillRect(box, QColor(235, 236, 238));
                    continue;
                }
                QColor fill = colour(cell.value);
                if (!cell.used) // not used: grey, with a trace of its colour
                    fill = QColor::fromRgbF(0.62 + 0.15 * fill.redF(), 0.62 + 0.15 * fill.greenF(), 0.62 + 0.15 * fill.blueF());
                painter.fillRect(box, fill);
                if (cell.negative) {
                    painter.setPen(Qt::NoPen);
                    painter.setBrush(Qt::black);
                    const double r = std::clamp(std::min(columnWidth, rowHeight) * 0.18, 1.0, 3.0);
                    painter.drawEllipse(box.center(), r, r);
                    painter.setBrush(Qt::NoBrush);
                }
            }
        painter.setPen(Qt::black);
        painter.drawRect(area);
        for (int g = 0; g < panel.gates; g += std::max(1, panel.gates / 5)) // gate numbers
            painter.drawText(QRectF(2, y + g * rowHeight, 66, rowHeight), Qt::AlignRight | Qt::AlignVCenter, QString::number(g + 1));
        if (m_active >= 0 && m_active < columns) {
            painter.setPen(QPen(QColor("#f28e2b"), 2.0));
            painter.drawRect(QRectF(left + m_active * columnWidth, y, columnWidth, area.height()));
        }
        y += area.height() + gap;
    }
    painter.save();
    painter.translate(18, (top + bottom) / 2);
    painter.rotate(-90);
    painter.setPen(Qt::black);
    painter.drawText(QRectF(-(bottom - top) / 2, -10, bottom - top, 20), Qt::AlignCenter, "Gate");
    painter.restore();

    // Strip: gates used (black) and RMS (red, right axis 0-3) per sounding.
    const QRectF strip(left, bottom + gap, right - left, stripHeight);
    painter.setPen(QColor(225, 228, 232));
    painter.drawLine(QPointF(strip.left(), strip.center().y()), QPointF(strip.right(), strip.center().y()));
    const int maximumGates = std::max(1, m_gatesUsed.isEmpty() ? 1 : *std::max_element(m_gatesUsed.begin(), m_gatesUsed.end()));
    QPolygonF gatesLine;
    for (int c = 0; c < std::min(columns, static_cast<int>(m_gatesUsed.size())); ++c)
        gatesLine << QPointF(left + (c + 0.5) * columnWidth, strip.bottom() - strip.height() * m_gatesUsed[c] / maximumGates);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(Qt::black, 1.5));
    painter.drawPolyline(gatesLine);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#d73027"));
    for (int c = 0; c < std::min(columns, static_cast<int>(m_rms.size())); ++c)
        if (std::isfinite(m_rms[c]))
            painter.drawEllipse(QPointF(left + (c + 0.5) * columnWidth, strip.bottom() - strip.height() * std::min(m_rms[c], 3.0) / 3.0), 2.5, 2.5);
    painter.setBrush(Qt::NoBrush);
    painter.setPen(Qt::black);
    painter.drawRect(strip);
    painter.drawText(QRectF(2, strip.top() - 8, 66, 16), Qt::AlignRight | Qt::AlignVCenter, QString::number(maximumGates));
    painter.drawText(QRectF(2, strip.bottom() - 8, 66, 16), Qt::AlignRight | Qt::AlignVCenter, "0");
    painter.drawText(QRectF(2, strip.center().y() - 8, 66, 16), Qt::AlignRight | Qt::AlignVCenter, "Gates");
    painter.setPen(QColor("#d73027"));
    painter.drawText(QRectF(strip.right() + 4, strip.top() - 8, 60, 16), Qt::AlignLeft | Qt::AlignVCenter, "RMS 3");
    painter.drawText(QRectF(strip.right() + 4, strip.bottom() - 8, 60, 16), Qt::AlignLeft | Qt::AlignVCenter, "0");
    // Distance labels.
    painter.setPen(Qt::black);
    for (int k = 0; k <= 4; ++k) {
        const int c = std::min(columns - 1, k * (columns - 1) / 4);
        painter.drawText(QRectF(left + (c + 0.5) * columnWidth - 40, strip.bottom() + 4, 80, 16), Qt::AlignHCenter,
                         QString::number(m_distances[c], 'f', 0));
    }
    painter.setFont(QFont(painter.font().family(), 9));
    painter.drawText(QRectF(left, height() - 22, right - left, 20), Qt::AlignCenter, "Distance [m]");

    // Colour bar, InverTEM's box style.
    const double barX = right + 58.0, barY = top + 22.0, barH = std::max(40.0, bottom - top - 30.0);
    painter.setPen(QColor(145, 150, 155));
    painter.setBrush(Qt::white);
    painter.drawRect(QRectF(right + 6.0, top - 4.0, 86.0, barH + 34.0));
    QLinearGradient gradient(0.0, barY + barH, 0.0, barY);
    for (int i = 0; i <= 16; ++i) {
        const double f = i / 16.0;
        gradient.setColorAt(f, colour(m_scale.logarithmic ? m_scale.minimum * std::pow(m_scale.maximum / m_scale.minimum, f)
                                                          : m_scale.minimum + f * (m_scale.maximum - m_scale.minimum)));
    }
    painter.setPen(Qt::black);
    painter.setBrush(gradient);
    painter.drawRect(QRectF(barX, barY, 13.0, barH));
    painter.setBrush(Qt::NoBrush);
    painter.setFont(QFont(painter.font().family(), 8));
    painter.drawText(QRectF(right + 6.0, top - 2.0, 86.0, 18.0), Qt::AlignCenter, m_scale.title);
    for (int tick = 0; tick <= 4; ++tick) {
        const double f = tick / 4.0;
        const double value = m_scale.logarithmic ? m_scale.minimum * std::pow(m_scale.maximum / m_scale.minimum, f)
                                                 : m_scale.minimum + f * (m_scale.maximum - m_scale.minimum);
        const double ty = barY + (1.0 - f) * barH;
        painter.drawLine(QPointF(barX - 4.0, ty), QPointF(barX, ty));
        painter.drawText(QRectF(barX - 52.0, ty - 8.0, 46.0, 16.0), Qt::AlignRight | Qt::AlignVCenter, QString::number(value, 'g', 2));
    }
}

bool PatchworkWidget::cellAt(const QPointF &point, int &panel, int &column, int &gate) const
{
    for (int p = 0; p < m_panelRects.size(); ++p) {
        const QRectF &area = m_panelRects[p];
        if (!area.contains(point) || m_distances.isEmpty())
            continue;
        panel = p;
        column = std::min(static_cast<int>(m_distances.size()) - 1, static_cast<int>((point.x() - area.left()) / area.width() * m_distances.size()));
        gate = std::min(std::max(1, m_panels[p].gates) - 1, static_cast<int>((point.y() - area.top()) / area.height() * std::max(1, m_panels[p].gates)));
        return true;
    }
    return false;
}

// Left click selects the sounding; right click toggles the gate.
void PatchworkWidget::mousePressEvent(QMouseEvent *event)
{
    int panel = 0, column = 0, gate = 0;
    if (!cellAt(event->position(), panel, column, gate))
        return;
    if (event->button() == Qt::LeftButton)
        emit columnClicked(column);
    else if (event->button() == Qt::RightButton)
        emit cellRightClicked(panel, column, gate);
}
