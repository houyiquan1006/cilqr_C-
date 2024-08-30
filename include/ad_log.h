#include <iostream>
#pragma once

#define AD_LERROR(X) std::cout << std::endl

#define AD_LDEBUG(X) AD_LERROR(X)
#define AD_LINFO(X) AD_LERROR(X)

#define AP_LERROR(X) AD_LERROR(X)
#define AP_LINFO(X) AD_LINFO(X)
#define AP_LDEBUG(X) AD_LDEBUG(X)
