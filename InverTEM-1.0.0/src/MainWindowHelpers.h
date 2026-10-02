// Helpers shared by MainWindow.cpp and Processing.cpp.
#pragma once

#include "UsfReader.h"

#include <QColor>

namespace invertem {

constexpr int transectPointIdOffset = 1000000; // point ids of the transect plots start here
bool validMapCoordinate(const pytem::UsfSounding &sounding);
double soundingDistanceMetres(const pytem::UsfSounding &first, const pytem::UsfSounding &second);
QColor momentColor(const std::string &name, std::size_t fallbackIndex);
int xyzLineNumber(const pytem::UsfSounding &sounding); // from "Line<n>_<k>"

} // namespace invertem
