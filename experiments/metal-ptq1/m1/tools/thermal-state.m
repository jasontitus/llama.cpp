#import <Foundation/Foundation.h>
int main(void) { @autoreleasepool { printf("%ld\n", (long)[NSProcessInfo processInfo].thermalState); } }
