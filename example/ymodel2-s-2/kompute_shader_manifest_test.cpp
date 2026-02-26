#include "../../include/kompute/kompute_shader_registry.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> required = {
        "broadcastInplace",
        "operator+","operator+=","operator-","operator-=","operator*","operator*=",
        "operator/","operator/=","operator%","operator%=","operator&","operator&=",
        "operator|","operator|=","operator^","operator^=","operator<<","operator<<=",
        "operator>>","operator>>=","operator<","operator<=","operator>","operator>=",
        "operator==","operator!=","matmul","matView","sum(int)","sum(vector)",
        "max(int)","max(vector)"
    };

    for (const auto& name : required) {
        if (!yt::kompute::hasYTensorBaseMathShader(name)) {
            std::cerr << "missing shader binding for: " << name << std::endl;
            return 1;
        }
    }

    std::cout << "kompute shader manifest ok: " << required.size() << " bindings" << std::endl;
    return 0;
}
