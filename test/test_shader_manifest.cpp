#include "../include/kompute/kompute_shader_registry.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> required = {
        "broadcastInplace",
        "operator+","operator+=","operator-","operator-=","operator*","operator*=",
        "operator/","operator/=","operator%(int)","operator%=(int)","operator&(int)","operator&=(int)",
        "operator|(int)","operator|=(int)","operator^(int)","operator^=(int)","operator<<(int)","operator<<=(int)",
        "operator>>(int)","operator>>=(int)","operator<","operator<=","operator>","operator>=",
        "operator==","operator!=","matmul","matView","sum(int)","sum(vector)",
        "max(int)","max(vector)"
    };

    int fail = 0;
    for (const auto& name : required) {
        if (!yt::kompute::hasYTensorBaseMathShader(name)) {
            std::cerr << "missing shader binding for: " << name << std::endl;
            ++fail;
        }
    }

    if (fail) {
        std::cerr << fail << " shader binding(s) missing" << std::endl;
        return 1;
    }

    std::cout << "kompute shader manifest ok: " << required.size() << " bindings" << std::endl;
    return 0;
}
