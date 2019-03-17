import math

class Compound:
    def __init__(self):
        pass
    
    def setPrinciple(self, principle):
        self.principle = principle

    def setCompundingRate(self, compounding_rate):
        self.compounding_rate = compounding_rate/100;

    def amountAfterYears(self, time):
        return self.principle * (1 + self.compounding_rate)**time

    def rule72period(self):
        return 72 / (self.compounding_rate*100)

    def doublingPeriod(self):
        return 1/math.log(1 + self.compounding_rate, 2.0)

    def discountingCashFlow(self, amount, time):
        return amount * (1 / (1 + self.compounding_rate))**time

    def pva(self, annuity, time):
        return annuity * self.pvaf(time)

    def pvaf(self, time):
        return (1 - 1/(1 + self.compounding_rate)**time)/self.compounding_rate

    def pvp(self, annuity):
        return annuity * self.pvpf()

    def pvpf(self):
        return 1 / self.compounding_rate

compound = Compound();
compound.setPrinciple(1000);
compound.setCompundingRate(10);
print(compound.amountAfterYears(10));
print(compound.rule72period());
print(compound.doublingPeriod());
print(compound.discountingCashFlow(1000, 6));
print(compound.pva(10, 8));
print(compound.pvp(1));