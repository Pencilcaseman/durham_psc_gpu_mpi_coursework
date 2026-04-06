#include <coursework/util.hpp>

namespace util {
    void exit_with(const char *msg) {
        std::fprintf( stderr, msg);
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}
