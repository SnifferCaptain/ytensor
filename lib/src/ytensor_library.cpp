#define YT_USE_LIB
#define YT_LIBRARY_IMPLEMENTATION
#include "../../ytensor.hpp"

namespace yt::infos {
std::unordered_map<std::string, yt::infos::TypeRegItem>& getTypeRegistry() {
    static std::unordered_map<std::string, yt::infos::TypeRegItem> registry;
    return registry;
}
}  // namespace yt::infos
