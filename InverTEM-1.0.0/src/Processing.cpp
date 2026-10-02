// Processing tools after EEMstudio (EEM Team, Milano): averaging along the
// lines, a noise model for the STDs, bulk culling, selections with keyboard
// commands and bookmarks.
#include "MainWindow.h"
#include "MainWindowHelpers.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

using namespace invertem;

namespace {

constexpr double mu0 = 4e-7 * 3.14159265358979323846;

bool isHighMoment(const std::string &name)
{
    const QString upper = QString::fromStdString(name).toUpper();
    return upper.startsWith("H") || upper.contains("HIGH");
}

double median(std::vector<double> values)
{
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

// Relative STD of a gate (1 when the value is zero).
double relativeStd(const pytem::UsfMoment &moment, std::size_t gate)
{
    const double value = std::abs(moment.voltages[gate]);
    return value > 0.0 && gate < moment.standardErrors.size() ? std::abs(moment.standardErrors[gate]) / value : 1.0;
}

// EEMstudio's weighted mean of a gate over the soundings of a window: weights
// 1/STD^2 over the used gates; values beyond two standard deviations of the
// mean are dropped and the mean recomputed, unless under 40 % lie within one.
// Returns the mean, its relative STD and whether any used gate took part.
struct Mean { double value, relativeStd; bool used; };
Mean weightedMean(const std::vector<double> &data, const std::vector<double> &relative, const std::vector<bool> &used)
{
    std::vector<std::size_t> kept;
    for (std::size_t i = 0; i < data.size(); ++i)
        if (used[i]) kept.push_back(i);
    const bool anyUsed = !kept.empty();
    if (data.size() == 1 || kept.size() == 1) {
        const std::size_t i = kept.empty() ? 0 : kept.front();
        return {data[i], relative[i], anyUsed};
    }
    if (kept.empty()) { // none used: all of them, unweighted spread
        kept.resize(data.size());
        std::iota(kept.begin(), kept.end(), 0);
    }
    const auto statistics = [&](const std::vector<std::size_t> &indices, double &mean, double &sigma, std::vector<double> &weights) {
        double sum = 0.0, sum2 = 0.0, weighted = 0.0;
        weights.assign(data.size(), 0.0);
        for (std::size_t i : indices) {
            const double absolute = std::max(1e-300, std::abs(data[i] * relative[i]));
            weights[i] = 1.0 / (absolute * absolute);
            sum += weights[i];
            sum2 += weights[i] * weights[i];
            weighted += weights[i] * data[i];
        }
        mean = weighted / sum;
        double variance = 0.0;
        for (std::size_t i : indices)
            variance += weights[i] * (data[i] - mean) * (data[i] - mean);
        const double denominator = sum - sum2 / sum;
        sigma = denominator > 0.0 ? std::sqrt(variance / denominator) : 0.0;
        return sum;
    };
    double mean = 0.0, sigma = 0.0;
    std::vector<double> weights;
    statistics(kept, mean, sigma, weights);
    if (!anyUsed) // all unused: the spread of one value
        return {mean, mean != 0.0 ? std::abs(sigma / mean) / std::sqrt(static_cast<double>(data.size())) : 0.0, false};
    std::vector<std::size_t> close;
    std::size_t within = 0;
    for (std::size_t i : kept) {
        const double misfit = sigma > 0.0 ? std::abs(data[i] - mean) / sigma : 0.0;
        within += misfit <= 1.0;
        if (misfit < 2.0) close.push_back(i);
    }
    if (static_cast<double>(within) / kept.size() >= 0.4 && !close.empty())
        kept = close;
    const double sum = statistics(kept, mean, sigma, weights);
    double normalised = 0.0;
    for (std::size_t i : kept)
        normalised += (weights[i] / sum) * (weights[i] / sum);
    return {mean, mean != 0.0 ? std::sqrt(normalised) * std::abs(sigma / mean) : 0.0, true};
}

} // namespace

double MainWindow::displayValue(const pytem::UsfSounding &sounding, double time, double value) const
{
    value = std::abs(value);
    if (!m_yUnit || m_yUnit->currentIndex() == 0 || !(time > 0.0) || !(value > 0.0))
        return value;
    // Late-time apparent resistivity of a loop of area A, from dB/dt per ampere.
    const double area = std::max(1e-6, sounding.loopX * sounding.loopY);
    return mu0 / (4.0 * 3.14159265358979323846 * time) * std::pow(2.0 * mu0 * area / (5.0 * time * value), 2.0 / 3.0);
}

// ── Averaging ─────────────────────────────────────────────────────────────

// Soundings of each line of XYZ data averaged in consecutive windows (time or
// distance along the line). Each gate averages the soundings within its own
// half-width of the window centre: the width grows with gate time (EEMstudio's
// trapezoid), interpolated in log time between three (time, width) points.
// The STD of an average adds the minimum STD in quadrature.
void MainWindow::averageSoundings()
{
    if (m_worker || m_soundings.empty())
        return;
    std::vector<std::string> momentNames;
    for (const auto &moment : m_soundings.front().sounding.moments)
        momentNames.push_back(moment.name);

    QDialog dialog(this);
    dialog.setWindowTitle("Average soundings");
    auto *layout = new QVBoxLayout(&dialog);
    layout->addWidget(new QLabel("Soundings of each line of XYZ data are averaged in windows along the line. Each gate\n"
                                 "averages the soundings within its own width of the window centre, so late gates average\n"
                                 "more soundings (weighted mean, outliers dropped). Widths between the three times are\n"
                                 "interpolated in log time; the first width of the first moment sets the window step."));
    auto *unitRow = new QHBoxLayout;
    auto *byTime = new QRadioButton("Time [s]"), *byDistance = new QRadioButton("Distance [m]");
    const bool haveTimes = std::all_of(m_soundings.begin(), m_soundings.end(), [](const SoundingState &s) {
        return !s.path.contains('#') || std::isfinite(s.sounding.time); });
    (haveTimes ? byTime : byDistance)->setChecked(true);
    byTime->setEnabled(haveTimes);
    unitRow->addWidget(new QLabel("Window unit:"));
    unitRow->addWidget(byTime);
    unitRow->addWidget(byDistance);
    unitRow->addStretch();
    layout->addLayout(unitRow);
    auto *table = new QTableWidget(static_cast<int>(momentNames.size()), 7);
    table->setHorizontalHeaderLabels({"Min. STD", "Time 1 [s]", "Time 2 [s]", "Time 3 [s]", "Width 1", "Width 2", "Width 3"});
    for (int row = 0; row < table->rowCount(); ++row) {
        const bool high = isHighMoment(momentNames[static_cast<std::size_t>(row)]);
        const double values[] = {0.03, high ? 1e-5 : 1e-6, high ? 1e-4 : 1e-5, high ? 1e-3 : 1e-4, 2.5, high ? 5.0 : 2.5, high ? 10.0 : 2.5};
        table->setVerticalHeaderItem(row, new QTableWidgetItem(QString::fromStdString(momentNames[static_cast<std::size_t>(row)])));
        for (int column = 0; column < 7; ++column)
            table->setItem(row, column, new QTableWidgetItem(QString::number(values[column], 'g', 3)));
    }
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->setMinimumWidth(620);
    table->setMaximumHeight(32 + 30 * table->rowCount());
    layout->addWidget(table);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttons->button(QDialogButtonBox::Ok)->setText("Average");
    QPushButton *restore = nullptr;
    if (!m_rawSoundings.empty())
        restore = buttons->addButton("Restore raw soundings", QDialogButtonBox::ResetRole);
    bool restoreRaw = false;
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (restore)
        connect(restore, &QPushButton::clicked, &dialog, [&] { restoreRaw = true; dialog.accept(); });
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted)
        return;
    if (restoreRaw) {
        m_soundings = std::move(m_rawSoundings);
        m_rawSoundings.clear();
        m_gateUndo.clear();
        m_selection.clear();
        m_log->appendPlainText(QString("Restored %1 raw soundings").arg(m_soundings.size()));
        markProjectModified();
        rebuildSoundingSelector(0);
        return;
    }
    struct Settings { double minStd; double times[3], halfWidths[3]; };
    std::map<std::string, Settings> settings;
    for (int row = 0; row < table->rowCount(); ++row) {
        Settings &s = settings[momentNames[static_cast<std::size_t>(row)]];
        s.minStd = table->item(row, 0)->text().toDouble();
        for (int k = 0; k < 3; ++k) {
            s.times[k] = std::max(1e-9, table->item(row, 1 + k)->text().toDouble());
            s.halfWidths[k] = std::max(0.0, table->item(row, 4 + k)->text().toDouble()) / 2.0;
        }
    }
    const auto halfWidth = [](const Settings &s, double time) {
        if (time <= s.times[0]) return s.halfWidths[0];
        if (time >= s.times[2]) return s.halfWidths[2];
        const int k = time < s.times[1] ? 0 : 1;
        const double f = std::log(time / s.times[k]) / std::log(s.times[k + 1] / s.times[k]);
        return s.halfWidths[k] + f * (s.halfWidths[k + 1] - s.halfWidths[k]);
    };
    const double step = 2.0 * settings[momentNames.front()].halfWidths[0];
    const bool distance = byDistance->isChecked();
    if (!(step > 0.0))
        return;

    std::vector<SoundingState> averaged;
    std::size_t lines = 0;
    for (std::size_t first = 0; first < m_soundings.size();) {
        // A run of soundings from one file and line; other soundings (USF) are kept as they are.
        const auto key = [&](std::size_t i) {
            return m_soundings[i].path.section('#', 0, 0) + QString::number(xyzLineNumber(m_soundings[i].sounding));
        };
        std::size_t last = first + 1;
        if (!m_soundings[first].path.contains('#')) {
            averaged.push_back(m_soundings[first]);
            first = last;
            continue;
        }
        while (last < m_soundings.size() && m_soundings[last].path.contains('#') && key(last) == key(first))
            ++last;
        ++lines;
        std::vector<double> position(last - first, 0.0);
        for (std::size_t i = first; i < last; ++i)
            position[i - first] = distance ? (i == first ? 0.0 : position[i - first - 1] + soundingDistanceMetres(m_soundings[i - 1].sounding, m_soundings[i].sounding))
                                           : m_soundings[i].sounding.time;
        const int line = xyzLineNumber(m_soundings[first].sounding);
        int number = 0;
        for (double start = position.front(); start <= position.back(); start += step) {
            const double centre = start + 0.5 * step;
            std::vector<std::size_t> members; // included soundings in the window
            for (std::size_t i = first; i < last; ++i)
                if (position[i - first] >= start && position[i - first] <= start + step && m_soundings[i].includeInBatch)
                    members.push_back(i);
            if (members.empty())
                continue;
            // The template: the member with the most moments, nearest the centre.
            const std::size_t nearest = *std::min_element(members.begin(), members.end(), [&](std::size_t a, std::size_t b) {
                const std::size_t ma = m_soundings[a].sounding.moments.size(), mb = m_soundings[b].sounding.moments.size();
                return ma != mb ? ma > mb : std::abs(position[a - first] - centre) < std::abs(position[b - first] - centre); });
            SoundingState state = m_soundings[nearest];
            pytem::UsfSounding &sounding = state.sounding;
            sounding.soundingNumber = ++number;
            sounding.soundingName = "Line" + std::to_string(line) + "_" + std::to_string(number);
            state.path = m_soundings[nearest].path.section('#', 0, 0) + '#' + QString::fromStdString(sounding.soundingName);
            state.haveResult = state.resultStale = false;
            state.savedModels.clear();
            state.result = {};
            state.originalErrors.clear();
            state.sources.clear();
            double lon = 0, lat = 0, x = 0, y = 0, z = 0, t = 0;
            for (std::size_t i : members) {
                const auto &member = m_soundings[i];
                lon += member.sounding.longitude; lat += member.sounding.latitude;
                x += member.sounding.sourceX; y += member.sounding.sourceY;
                z += member.sounding.elevation; t += member.sounding.time;
                state.bookmark = state.bookmark || member.bookmark;
                state.sources.push_back(static_cast<int>(i));
            }
            const double n = static_cast<double>(members.size());
            sounding.longitude = lon / n; sounding.latitude = lat / n;
            sounding.sourceX = x / n; sounding.sourceY = y / n;
            sounding.elevation = z / n; sounding.time = t / n;
            for (std::size_t m = 0; m < sounding.moments.size(); ++m) {
                auto &moment = sounding.moments[m];
                const auto found = settings.find(moment.name);
                const Settings &s = found != settings.end() ? found->second : settings.begin()->second;
                moment.stackCount = 0;
                for (std::size_t gate = 0; gate < moment.times.size(); ++gate) {
                    const double reach = halfWidth(s, moment.times[gate]);
                    std::vector<double> values, relative;
                    std::vector<bool> used;
                    for (std::size_t i = first; i < last; ++i) {
                        const auto &member = m_soundings[i];
                        if (std::abs(position[i - first] - centre) > reach || !member.includeInBatch)
                            continue;
                        const auto &others = member.sounding.moments;
                        const auto other = std::find_if(others.begin(), others.end(), [&](const pytem::UsfMoment &o) { return o.name == moment.name; });
                        if (other == others.end() || gate >= other->voltages.size())
                            continue;
                        values.push_back(other->voltages[gate]);
                        relative.push_back(relativeStd(*other, gate));
                        used.push_back(member.gateEnabled[static_cast<std::size_t>(other - others.begin())][gate]);
                        if (gate == 0)
                            moment.stackCount += std::max(1, other->stackCount);
                    }
                    if (values.empty())
                        continue;
                    const Mean mean = weightedMean(values, relative, used);
                    moment.voltages[gate] = mean.value;
                    moment.standardErrors[gate] = std::hypot(mean.relativeStd, s.minStd) * std::abs(mean.value);
                    moment.qualityAccepted[gate] = true;
                    state.gateEnabled[m][gate] = mean.used;
                }
            }
            averaged.push_back(std::move(state));
        }
        first = last;
    }
    const std::size_t before = m_soundings.size();
    m_rawSoundings = std::move(m_soundings);
    m_soundings = std::move(averaged);
    m_gateUndo.clear();
    m_selection.clear();
    m_transectStart = 0;
    m_log->appendPlainText(QString("Averaged %1 soundings on %2 lines into %3 (windows of %4 %5); "
                                   "the raw soundings can be restored from this dialog until InverTEM is closed")
                               .arg(before).arg(lines).arg(m_soundings.size()).arg(step).arg(distance ? "m" : "s"));
    markProjectModified();
    rebuildSoundingSelector(0);
}

// ── Noise model and STD edits ────────────────────────────────────────────

// STD = sqrt(uniform^2 + (noise / |d|)^2) per gate, with the noise N (dB/dt
// [V/m^2] at 1 ms, divided by the current) falling as t^-1/2 (EEMverter's
// InductiveNoiseLevel model). The base is a uniform relative STD or the data's
// own STDs; the original STDs are kept for a reset. The window stays open, so
// the levels can be tuned while browsing the soundings.
void MainWindow::editNoiseModel()
{
    if (auto *open = findChild<QDialog *>("noiseModel")) {
        open->raise();
        return;
    }
    auto *dialog = new QDialog(this);
    dialog->setObjectName("noiseModel");
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle("Noise model");
    auto *form = new QFormLayout(dialog);
    form->addRow(new QLabel("STD = √(base² + (N / I · (t / 1 ms)^-½ / |dB/dt|)²) per gate; N is the noise at 1 ms\n"
                            "in V/m², dashed on the decay plot. This window stays open: apply, browse, adjust."));
    auto *uniform = new QRadioButton("Uniform relative STD"), *own = new QRadioButton("The data's own STDs");
    uniform->setChecked(true);
    auto *uniformStd = new QDoubleSpinBox;
    uniformStd->setRange(0.1, 100.0);
    uniformStd->setSuffix(" %");
    uniformStd->setValue(100.0 * m_noiseUniformStd);
    auto *base = new QHBoxLayout;
    base->addWidget(uniform);
    base->addWidget(uniformStd);
    base->addWidget(own);
    form->addRow("Base", base);
    // N of a moment from the data's own STDs: over every gate noisier than the uniform
    // STD, the median of what the uniform STD leaves, scaled to 1 A and 1 ms.
    const auto estimate = [=](const std::string &name) {
        std::vector<double> noise;
        for (const auto &state : m_soundings)
            for (std::size_t m = 0; m < state.sounding.moments.size(); ++m) {
                const auto &moment = state.sounding.moments[m];
                const auto &errors = m < state.originalErrors.size() ? state.originalErrors[m] : moment.standardErrors;
                for (std::size_t gate = 0; moment.name == name && gate < moment.times.size(); ++gate) {
                    const double excess = std::pow(errors[gate], 2) - std::pow(uniformStd->value() / 100.0 * moment.voltages[gate], 2);
                    if (moment.voltages[gate] != 0.0 && excess > 0.0)
                        noise.push_back(std::sqrt(excess) * moment.meanCurrent * std::sqrt(moment.times[gate] / 1e-3));
                }
            }
        return noise.empty() ? 1e-9 : median(noise);
    };
    std::map<std::string, QLineEdit *> levels;
    for (const auto &state : m_soundings)
        for (const auto &moment : state.sounding.moments)
            if (!levels.count(moment.name)) {
                const auto found = m_noiseLevel.find(moment.name);
                auto *edit = new QLineEdit(QString::number(found != m_noiseLevel.end() ? found->second : estimate(moment.name), 'g', 2));
                form->addRow(QString("%1 noise at 1 ms [V/m²]").arg(QString::fromStdString(moment.name)), edit);
                levels[moment.name] = edit;
            }
    const auto apply = [=](bool reset) {
        if (m_worker || m_soundings.empty())
            return;
        m_noiseLevel.clear();
        QStringList summary{uniform->isChecked() ? QString("base %1 %").arg(uniformStd->value()) : "base the data's STDs"};
        if (!reset) {
            m_noiseUniformStd = uniformStd->value() / 100.0;
            for (const auto &[name, edit] : levels) {
                m_noiseLevel[name] = std::max(0.0, edit->text().replace(QLatin1Char(','), QLatin1Char('.')).toDouble());
                summary << QString("%1 %2 V/m²").arg(QString::fromStdString(name)).arg(m_noiseLevel[name], 0, 'g', 2);
            }
        }
        std::vector<double> before, after; // relative STDs of the used gates
        for (auto &state : m_soundings) {
            auto &moments = state.sounding.moments;
            if (state.originalErrors.empty())
                for (const auto &moment : moments)
                    state.originalErrors.push_back(moment.standardErrors);
            for (std::size_t m = 0; m < moments.size() && m < state.originalErrors.size(); ++m) {
                auto &moment = moments[m];
                for (std::size_t gate = 0; gate < moment.times.size(); ++gate) {
                    const double value = std::abs(moment.voltages[gate]), original = std::abs(state.originalErrors[m][gate]);
                    if (reset || !(value > 0.0)) {
                        moment.standardErrors[gate] = original;
                        continue;
                    }
                    const double noise = m_noiseLevel[moment.name] / std::max(1e-12, moment.meanCurrent)
                        * std::pow(moment.times[gate] / 1e-3, -0.5);
                    moment.standardErrors[gate] = std::hypot(uniform->isChecked() ? m_noiseUniformStd * value : original, noise);
                    if (state.includeInBatch && state.gateEnabled[m][gate]) {
                        before.push_back(original / value);
                        after.push_back(moment.standardErrors[gate] / value);
                    }
                }
            }
            if (reset)
                state.originalErrors.clear();
            state.resultStale = state.haveResult;
        }
        m_log->appendPlainText(reset ? QString("STDs reset to the data's own") : QString("Noise model applied (%1)").arg(summary.join(", "))
            + (before.empty() ? QString() : QString(": median STD of the used gates %1 % -> %2 %; the inversion uses at least the error floor")
                  .arg(100.0 * median(before), 0, 'f', 1).arg(100.0 * median(after), 0, 'f', 1)));
        refreshAfterEdit(true);
    };
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Close);
    auto *estimateButton = buttons->addButton("Estimate from data", QDialogButtonBox::ActionRole);
    estimateButton->setToolTip("Set the noise levels from the data's own STDs (median over all soundings)");
    auto *resetButton = buttons->addButton("Reset to original STDs", QDialogButtonBox::ResetRole);
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked, dialog, [=] { apply(false); });
    connect(resetButton, &QPushButton::clicked, dialog, [=] { apply(true); });
    connect(estimateButton, &QPushButton::clicked, dialog, [=] {
        for (const auto &[name, edit] : levels)
            edit->setText(QString::number(estimate(name), 'g', 2));
    });
    connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::close);
    form->addRow(buttons);
    dialog->show();
}

void MainWindow::removeHighStd()
{
    if (m_worker || m_soundings.empty())
        return;
    bool ok = false;
    const double limit = QInputDialog::getDouble(this, "Remove gates with a high STD", "Remove used gates with a relative STD of at least [%]:",
                                                 50.0, 1.0, 1000.0, 0, &ok) / 100.0;
    if (!ok)
        return;
    std::vector<GateReference> references;
    for (std::size_t s = 0; s < m_soundings.size(); ++s) {
        const auto &state = m_soundings[s];
        for (std::size_t m = 0; state.includeInBatch && m < state.sounding.moments.size(); ++m)
            for (std::size_t gate = 0; gate < state.sounding.moments[m].times.size(); ++gate)
                if (state.gateEnabled[m][gate] && relativeStd(state.sounding.moments[m], gate) >= limit)
                    references.push_back({static_cast<int>(s), m, gate});
    }
    m_log->appendPlainText(QString("Removed %1 gates with a relative STD of %2 % or more").arg(references.size()).arg(100.0 * limit));
    editReferences(references, GateEdit::Remove);
}

void MainWindow::selectPoints(const QVector<int> &pointIds)
{
    m_selection = resolvePoints(pointIds);
    m_log->appendPlainText(QString("%1 gates selected: Q remove, A restore, N remove negatives, E/W remove after/before "
                                   "(Shift: the whole page), 1/2/5 add 10/20/50 % STD, 0 original STD, Esc clear")
                               .arg(m_selection.size()));
    refreshAfterEdit(false);
}

// A command on the selection; E and W cut every sounding of the selection (or,
// with Shift, of the transect page) from the first selected gate to the last
// gate, or from the first gate to the last selected one, per moment.
void MainWindow::selectionCommand(char key, bool wholePage)
{
    if (m_worker)
        return;
    if (key == 27) { // Esc
        m_selection.clear();
        refreshAfterEdit(false);
        return;
    }
    if (m_selection.empty())
        return;
    if (key == 'Q' || key == 'A') {
        editReferences(m_selection, key == 'Q' ? GateEdit::Remove : GateEdit::Restore);
        return;
    }
    std::vector<GateReference> references;
    if (key == 'N') {
        for (const auto &reference : m_selection)
            if (m_soundings[static_cast<std::size_t>(reference.sounding)].sounding.moments[reference.moment].voltages[reference.gate] < 0.0)
                references.push_back(reference);
    } else if (key == 'E' || key == 'W') {
        std::map<std::size_t, std::size_t> cut; // moment: first (E) or last (W) selected gate
        std::vector<int> soundings;
        for (const auto &reference : m_selection) {
            auto [entry, added] = cut.emplace(reference.moment, reference.gate);
            if (!added)
                entry->second = key == 'E' ? std::min(entry->second, reference.gate) : std::max(entry->second, reference.gate);
            soundings.push_back(reference.sounding);
        }
        if (wholePage) {
            soundings = m_plotMode->currentIndex() == 1 ? transectPage() : std::vector<int>{m_activeSoundingIndex};
        }
        std::sort(soundings.begin(), soundings.end());
        soundings.erase(std::unique(soundings.begin(), soundings.end()), soundings.end());
        for (int s : soundings) {
            const auto &state = m_soundings[static_cast<std::size_t>(s)];
            for (const auto &[moment, gate] : cut) {
                if (!state.includeInBatch || moment >= state.gateEnabled.size())
                    continue;
                const std::size_t gates = state.gateEnabled[moment].size();
                for (std::size_t g = key == 'E' ? gate : 0; g < (key == 'E' ? gates : std::min(gates, gate + 1)); ++g)
                    references.push_back({s, moment, g});
            }
        }
    } else {
        return;
    }
    editReferences(references, GateEdit::Remove);
}

void MainWindow::changeSelectionStd(double addRelative)
{
    if (m_worker || m_selection.empty())
        return;
    bool activeChanged = false;
    for (const auto &reference : m_selection) {
        auto &state = m_soundings[static_cast<std::size_t>(reference.sounding)];
        auto &moment = state.sounding.moments[reference.moment];
        if (state.originalErrors.empty())
            for (const auto &each : state.sounding.moments)
                state.originalErrors.push_back(each.standardErrors);
        moment.standardErrors[reference.gate] = addRelative < 0.0
            ? state.originalErrors[reference.moment][reference.gate]
            : (relativeStd(moment, reference.gate) + addRelative) * std::abs(moment.voltages[reference.gate]);
        state.resultStale = state.haveResult;
        activeChanged = activeChanged || reference.sounding == m_activeSoundingIndex;
    }
    m_log->appendPlainText(addRelative < 0.0 ? QString("Original STDs of %1 gates restored").arg(m_selection.size())
                                             : QString("STD of %1 gates raised by %2 %").arg(m_selection.size()).arg(100.0 * addRelative));
    refreshAfterEdit(activeChanged);
}

// ── Bookmarks ─────────────────────────────────────────────────────────────

void MainWindow::toggleBookmark()
{
    auto *state = activeSounding();
    if (!state)
        return;
    state->bookmark = !state->bookmark;
    const QString text = m_soundingSelector->itemText(m_activeSoundingIndex);
    m_soundingSelector->setItemText(m_activeSoundingIndex, state->bookmark ? "★ " + text : text.mid(2));
    markProjectModified();
    updateMapPlot();
}

void MainWindow::nextBookmark()
{
    const int count = static_cast<int>(m_soundings.size());
    for (int step = 1; step <= count; ++step) {
        const int index = (m_activeSoundingIndex + step) % count;
        if (m_soundings[static_cast<std::size_t>(index)].bookmark) {
            navigateToSounding(index);
            return;
        }
    }
}
