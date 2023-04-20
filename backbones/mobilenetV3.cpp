#include <iostream>

int make_divisible(int v,int divisor,int min_value){
    int new_value = (v + divisor / 2) / divisor * divisor;
    new_value = new_value >= min_value ? new_value:min_value;
    if(static_cast<float>(new_value) < static_cast<float>(v) * 0.9f) {
        new_value += divisor;
    } 
    return new_value;
}

// test the divisible function
int main() {
    int inplanes = 16;
    int output_channels = make_divisible(inplanes,8,8);
    std::printf("the inplanes is %d,the output channles is %d\n",inplanes,output_channels);
}