from collections import defaultdict
import re


class TreeNode(object):
    def __init__(self, text, offset, elements=None):
        self.text = text
        self.offset = offset
        self.elements = elements or []

    def __iter__(self):
        for el in self.elements:
            yield el


class TreeNode1(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode1, self).__init__(text, offset, elements)
        self.BooleanValue = elements[1]
        self.EndOfInput = elements[2]


class TreeNode2(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode2, self).__init__(text, offset, elements)
        self.BooleanValue = elements[1]
        self.EndOfInput = elements[2]


class TreeNode3(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode3, self).__init__(text, offset, elements)
        self.NumericValue = elements[1]
        self.EndOfInput = elements[2]


class TreeNode4(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode4, self).__init__(text, offset, elements)
        self.StringValue = elements[1]
        self.EndOfInput = elements[2]


class TreeNode5(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode5, self).__init__(text, offset, elements)
        self.EndOfInput = elements[2]


class TreeNode6(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode6, self).__init__(text, offset, elements)
        self.BooleanValue = elements[0]


class TreeNode7(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode7, self).__init__(text, offset, elements)
        self.BooleanValue = elements[0]


class TreeNode8(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode8, self).__init__(text, offset, elements)
        self.OrFactor = elements[0]


class TreeNode9(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode9, self).__init__(text, offset, elements)
        self.OrFactor = elements[3]


class TreeNode10(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode10, self).__init__(text, offset, elements)
        self.BoolFactor = elements[0]


class TreeNode11(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode11, self).__init__(text, offset, elements)
        self.BoolFactor = elements[3]


class TreeNode12(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode12, self).__init__(text, offset, elements)
        self.BoolFactor = elements[3]


class TreeNode13(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode13, self).__init__(text, offset, elements)
        self.BooleanValue = elements[2]


class TreeNode14(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode14, self).__init__(text, offset, elements)
        self.NumericValue = elements[0]
        self.NumericComparison = elements[1]


class TreeNode15(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode15, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode16(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode16, self).__init__(text, offset, elements)
        self.StringValue = elements[0]
        self.StringComparison = elements[1]


class TreeNode17(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode17, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode18(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode18, self).__init__(text, offset, elements)
        self.AnyValueParen = elements[2]


class TreeNode19(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode19, self).__init__(text, offset, elements)
        self.AnyValueParen = elements[2]


class TreeNode20(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode20, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.StringConst = elements[5]


class TreeNode21(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode21, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.StringConst = elements[5]


class TreeNode22(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode22, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.StringConst = elements[5]


class TreeNode23(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode23, self).__init__(text, offset, elements)
        self.NumericOrStringValueComma = elements[2]
        self.ListValue = elements[5]


class TreeNode24(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode24, self).__init__(text, offset, elements)
        self.BooleanValue = elements[2]


class TreeNode25(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode25, self).__init__(text, offset, elements)
        self.BooleanValue = elements[2]


class TreeNode26(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode26, self).__init__(text, offset, elements)
        self.NumericValue = elements[0]


class TreeNode27(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode27, self).__init__(text, offset, elements)
        self.NumericValue = elements[0]


class TreeNode28(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode28, self).__init__(text, offset, elements)
        self.AddFactor = elements[0]


class TreeNode29(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode29, self).__init__(text, offset, elements)
        self.AddFactor = elements[2]


class TreeNode30(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode30, self).__init__(text, offset, elements)
        self.NumericFactor = elements[0]


class TreeNode31(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode31, self).__init__(text, offset, elements)
        self.NumericFactor = elements[2]


class TreeNode32(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode32, self).__init__(text, offset, elements)
        self.NumericFactor = elements[2]


class TreeNode33(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode33, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode34(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode34, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode35(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode35, self).__init__(text, offset, elements)
        self.NumericValue = elements[5]


class TreeNode36(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode36, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]
        self.IntegerConst = elements[5]


class TreeNode37(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode37, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode38(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode38, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.StringConst = elements[5]


class TreeNode39(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode39, self).__init__(text, offset, elements)
        self.NumericOrStringValueParen = elements[2]


class TreeNode40(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode40, self).__init__(text, offset, elements)
        self.NumericValue = elements[5]


class TreeNode41(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode41, self).__init__(text, offset, elements)
        self.NumericValueParen = elements[3]


class TreeNode42(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode42, self).__init__(text, offset, elements)
        self.NumericValueParen = elements[3]


class TreeNode43(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode43, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode44(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode44, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode45(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode45, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode46(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode46, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode47(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode47, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode48(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode48, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode49(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode49, self).__init__(text, offset, elements)
        self.NumericValue = elements[2]


class TreeNode50(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode50, self).__init__(text, offset, elements)
        self.NumericOrStringValueParen = elements[2]


class TreeNode51(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode51, self).__init__(text, offset, elements)
        self.NumericOrStringValueParen = elements[2]


class TreeNode52(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode52, self).__init__(text, offset, elements)
        self.NumericValueParen = elements[3]


class TreeNode53(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode53, self).__init__(text, offset, elements)
        self.NumericValueComma = elements[0]


class TreeNode54(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode54, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode55(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode55, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode56(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode56, self).__init__(text, offset, elements)
        self.SeriesGroupBy = elements[2]


class TreeNode57(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode57, self).__init__(text, offset, elements)
        self.SeriesGroupBy = elements[2]


class TreeNode58(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode58, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode59(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode59, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode60(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode60, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode61(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode61, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode62(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode62, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode63(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode63, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode64(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode64, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode65(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode65, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode66(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode66, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode67(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode67, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode68(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode68, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode69(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode69, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode70(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode70, self).__init__(text, offset, elements)
        self.GroupBySeriesParam = elements[2]


class TreeNode71(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode71, self).__init__(text, offset, elements)
        self.DataFrameGroupBy = elements[2]
        self.ExpressionParam = elements[5]


class TreeNode72(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode72, self).__init__(text, offset, elements)
        self.DataFrameGroupBy = elements[0]
        self.VariableName = elements[3]


class TreeNode73(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode73, self).__init__(text, offset, elements)
        self.AnyValueComma = elements[2]
        self.AnyValueParen = elements[7]


class TreeNode74(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode74, self).__init__(text, offset, elements)
        self.ExpressionConst = elements[2]


class TreeNode75(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode75, self).__init__(text, offset, elements)
        self.StringValue = elements[0]


class TreeNode76(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode76, self).__init__(text, offset, elements)
        self.StringValue = elements[0]


class TreeNode77(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode77, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode78(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode78, self).__init__(text, offset, elements)
        self.StringValue = elements[5]


class TreeNode79(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode79, self).__init__(text, offset, elements)
        self.StringValue = elements[8]
        self.NumericValue = elements[5]


class TreeNode80(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode80, self).__init__(text, offset, elements)
        self.StringValue = elements[8]
        self.NumericValue = elements[5]


class TreeNode81(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode81, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode82(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode82, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode83(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode83, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode84(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode84, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode85(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode85, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode86(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode86, self).__init__(text, offset, elements)
        self.StringValue = elements[2]


class TreeNode87(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode87, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.StringConst = elements[8]


class TreeNode88(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode88, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.IntegerConst = elements[5]


class TreeNode89(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode89, self).__init__(text, offset, elements)
        self.IntegerConst = elements[2]


class TreeNode90(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode90, self).__init__(text, offset, elements)
        self.StringValue = elements[2]
        self.IntegerConst = elements[5]


class TreeNode91(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode91, self).__init__(text, offset, elements)
        self.NumericOrStringValueParen = elements[2]


class TreeNode92(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode92, self).__init__(text, offset, elements)
        self.StringValue = elements[5]


class TreeNode93(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode93, self).__init__(text, offset, elements)
        self.BooleanValue = elements[0]


class TreeNode94(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode94, self).__init__(text, offset, elements)
        self.NumericValue = elements[0]


class TreeNode95(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode95, self).__init__(text, offset, elements)
        self.StringValue = elements[0]


class TreeNode96(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode96, self).__init__(text, offset, elements)
        self.BooleanValue = elements[0]


class TreeNode97(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode97, self).__init__(text, offset, elements)
        self.BooleanValue = elements[0]


class TreeNode98(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode98, self).__init__(text, offset, elements)
        self.AnyValue = elements[0]


class TreeNode99(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode99, self).__init__(text, offset, elements)
        self.AnyValue = elements[2]


class TreeNode100(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode100, self).__init__(text, offset, elements)
        self.BooleanValueComma = elements[2]
        self.AnyValueComma = elements[5]
        self.AnyValueParen = elements[8]


class TreeNode101(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode101, self).__init__(text, offset, elements)
        self.BooleanValue = elements[2]
        self.Then = elements[3]
        self.AnyValueElse = elements[4]
        self.Else = elements[6]
        self.AnyValueParen = elements[7]


class TreeNode102(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode102, self).__init__(text, offset, elements)
        self.Elseif = elements[0]
        self.BooleanValue = elements[1]
        self.Then = elements[2]
        self.AnyValueElse = elements[3]


class TreeNode103(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode103, self).__init__(text, offset, elements)
        self.AnyValueComma = elements[2]
        self.BooleanConst = elements[5]


class TreeNode104(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode104, self).__init__(text, offset, elements)
        self.FalseConst = elements[0]


class TreeNode105(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode105, self).__init__(text, offset, elements)
        self.StringConst = elements[0]


class TreeNode106(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode106, self).__init__(text, offset, elements)
        self.VariableName = elements[2]


class TreeNode107(TreeNode):
    def __init__(self, text, offset, elements):
        super(TreeNode107, self).__init__(text, offset, elements)
        self.VariableName = elements[2]


class ParseError(SyntaxError):
    pass


FAILURE = object()


class Grammar(object):
    REGEX_1 = re.compile('^[0-9]')
    REGEX_2 = re.compile('^[0-9]')
    REGEX_3 = re.compile('^[0-9]')
    REGEX_4 = re.compile('^[^\']')
    REGEX_5 = re.compile('^[^"]')
    REGEX_6 = re.compile('^[a-zA-Z0-9_]')
    REGEX_7 = re.compile('^[ ]')
    REGEX_8 = re.compile('^[^\']')
    REGEX_9 = re.compile('^[^"]')
    REGEX_10 = re.compile('^[a-zA-Z_]')

    def _read_ChecksExpression(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ChecksExpression'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        remaining0, index2, elements1, address2 = 0, self._offset, [], True
        while address2 is not FAILURE:
            address2 = self._read_WS()
            if address2 is not FAILURE:
                elements1.append(address2)
                remaining0 -= 1
        if remaining0 <= 0:
            address1 = TreeNode(self._input[index2:self._offset], index2, elements1)
            self._offset = self._offset
        else:
            address1 = FAILURE
        if address1 is not FAILURE:
            elements0.append(address1)
            address3 = FAILURE
            address3 = self._read_BooleanValue()
            if address3 is not FAILURE:
                elements0.append(address3)
                address4 = FAILURE
                address4 = self._read_EndOfInput()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.checks_expression(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ChecksExpression'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanExpression(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanExpression'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        remaining0, index2, elements1, address2 = 0, self._offset, [], True
        while address2 is not FAILURE:
            address2 = self._read_WS()
            if address2 is not FAILURE:
                elements1.append(address2)
                remaining0 -= 1
        if remaining0 <= 0:
            address1 = TreeNode(self._input[index2:self._offset], index2, elements1)
            self._offset = self._offset
        else:
            address1 = FAILURE
        if address1 is not FAILURE:
            elements0.append(address1)
            address3 = FAILURE
            address3 = self._read_BooleanValue()
            if address3 is not FAILURE:
                elements0.append(address3)
                address4 = FAILURE
                address4 = self._read_EndOfInput()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.boolean_expression(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['BooleanExpression'][index0] = (address0, self._offset)
        return address0

    def _read_NumericExpression(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericExpression'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        remaining0, index2, elements1, address2 = 0, self._offset, [], True
        while address2 is not FAILURE:
            address2 = self._read_WS()
            if address2 is not FAILURE:
                elements1.append(address2)
                remaining0 -= 1
        if remaining0 <= 0:
            address1 = TreeNode(self._input[index2:self._offset], index2, elements1)
            self._offset = self._offset
        else:
            address1 = FAILURE
        if address1 is not FAILURE:
            elements0.append(address1)
            address3 = FAILURE
            address3 = self._read_NumericValue()
            if address3 is not FAILURE:
                elements0.append(address3)
                address4 = FAILURE
                address4 = self._read_EndOfInput()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.numeric_expression(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NumericExpression'][index0] = (address0, self._offset)
        return address0

    def _read_StringExpression(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringExpression'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        remaining0, index2, elements1, address2 = 0, self._offset, [], True
        while address2 is not FAILURE:
            address2 = self._read_WS()
            if address2 is not FAILURE:
                elements1.append(address2)
                remaining0 -= 1
        if remaining0 <= 0:
            address1 = TreeNode(self._input[index2:self._offset], index2, elements1)
            self._offset = self._offset
        else:
            address1 = FAILURE
        if address1 is not FAILURE:
            elements0.append(address1)
            address3 = FAILURE
            address3 = self._read_StringValue()
            if address3 is not FAILURE:
                elements0.append(address3)
                address4 = FAILURE
                address4 = self._read_EndOfInput()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.string_expression(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StringExpression'][index0] = (address0, self._offset)
        return address0

    def _read_AnyExpression(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyExpression'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        remaining0, index2, elements1, address2 = 0, self._offset, [], True
        while address2 is not FAILURE:
            address2 = self._read_WS()
            if address2 is not FAILURE:
                elements1.append(address2)
                remaining0 -= 1
        if remaining0 <= 0:
            address1 = TreeNode(self._input[index2:self._offset], index2, elements1)
            self._offset = self._offset
        else:
            address1 = FAILURE
        if address1 is not FAILURE:
            elements0.append(address1)
            address3 = FAILURE
            index3 = self._offset
            address3 = self._read_BooleanExpression()
            if address3 is FAILURE:
                self._offset = index3
                address3 = self._read_NumericExpression()
                if address3 is FAILURE:
                    self._offset = index3
                    address3 = self._read_StringExpression()
                    if address3 is FAILURE:
                        self._offset = index3
            if address3 is not FAILURE:
                elements0.append(address3)
                address4 = FAILURE
                address4 = self._read_EndOfInput()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode5(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['AnyExpression'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanValueComma(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanValueComma'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BooleanValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode6(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['BooleanValueComma'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanValueParen(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanValueParen'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BooleanValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ')':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\')\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode7(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['BooleanValueParen'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_OrFactor()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_OrExpr()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode8(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['BooleanValue'][index0] = (address0, self._offset)
        return address0

    def _read_OrExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['OrExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 2]
        if chunk0 == 'or':
            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
            self._offset = self._offset + 2
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'or\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 3]
            if chunk1 == 'xor':
                address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
                self._offset = self._offset + 3
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'xor\'')
            if address1 is FAILURE:
                self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            address2 = self._read_ANC()
            self._offset = index3
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index4, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index4:self._offset], index4, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address5 = FAILURE
                    address5 = self._read_OrFactor()
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.or_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['OrExpr'][index0] = (address0, self._offset)
        return address0

    def _read_OrFactor(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['OrFactor'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BoolFactor()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_AndExpr()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode10(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['OrFactor'][index0] = (address0, self._offset)
        return address0

    def _read_AndExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AndExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 3]
        if chunk0 == 'and':
            address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'and\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address5 = FAILURE
                    address5 = self._read_BoolFactor()
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.and_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['AndExpr'][index0] = (address0, self._offset)
        return address0

    def _read_NotExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NotExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 3]
        if chunk0 == 'not':
            address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'not\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address5 = FAILURE
                    address5 = self._read_BoolFactor()
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.not_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NotExpr'][index0] = (address0, self._offset)
        return address0

    def _read_BoolFactor(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BoolFactor'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3, elements1 = self._offset, []
        address2 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '(':
            address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address2 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'(\'')
        if address2 is not FAILURE:
            elements1.append(address2)
            address3 = FAILURE
            remaining0, index4, elements2, address4 = 0, self._offset, [], True
            while address4 is not FAILURE:
                address4 = self._read_WS()
                if address4 is not FAILURE:
                    elements2.append(address4)
                    remaining0 -= 1
            if remaining0 <= 0:
                address3 = TreeNode(self._input[index4:self._offset], index4, elements2)
                self._offset = self._offset
            else:
                address3 = FAILURE
            if address3 is not FAILURE:
                elements1.append(address3)
                address5 = FAILURE
                address5 = self._read_BooleanValue()
                if address5 is not FAILURE:
                    elements1.append(address5)
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address6 is not FAILURE:
                        elements1.append(address6)
                    else:
                        elements1 = None
                        self._offset = index3
                else:
                    elements1 = None
                    self._offset = index3
            else:
                elements1 = None
                self._offset = index3
        else:
            elements1 = None
            self._offset = index3
        if elements1 is None:
            address1 = FAILURE
        else:
            address1 = TreeNode13(self._input[index3:self._offset], index3, elements1)
            self._offset = self._offset
        if address1 is FAILURE:
            self._offset = index2
            address1 = self._read_BooleanExpr()
            if address1 is FAILURE:
                self._offset = index2
                address1 = self._read_BooleanFunctions()
                if address1 is FAILURE:
                    self._offset = index2
                    address1 = self._read_BoolFactor2()
                    if address1 is FAILURE:
                        self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address7 = FAILURE
            remaining1, index5, elements3, address8 = 0, self._offset, [], True
            while address8 is not FAILURE:
                address8 = self._read_WS()
                if address8 is not FAILURE:
                    elements3.append(address8)
                    remaining1 -= 1
            if remaining1 <= 0:
                address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                self._offset = self._offset
            else:
                address7 = FAILURE
            if address7 is not FAILURE:
                elements0.append(address7)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['BoolFactor'][index0] = (address0, self._offset)
        return address0

    def _read_BoolFactor2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BoolFactor2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_BooleanConst()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NoneConst()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_Variable()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_NotExpr()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['BoolFactor2'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_NumericCompExpr()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_StringCompExpr()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['BooleanExpr'][index0] = (address0, self._offset)
        return address0

    def _read_NumericCompExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericCompExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_NumericValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            address2 = self._read_NumericComparison()
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode14(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['NumericCompExpr'][index0] = (address0, self._offset)
        return address0

    def _read_NumericComparison(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericComparison'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 2]
        if chunk0 == '<=':
            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
            self._offset = self._offset + 2
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'<=\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 1]
            if chunk1 == '<':
                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'<\'')
            if address1 is FAILURE:
                self._offset = index2
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 2]
                if chunk2 == '>=':
                    address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                    self._offset = self._offset + 2
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'>=\'')
                if address1 is FAILURE:
                    self._offset = index2
                    chunk3 = None
                    if self._offset < self._input_size:
                        chunk3 = self._input[self._offset:self._offset + 1]
                    if chunk3 == '>':
                        address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address1 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\'>\'')
                    if address1 is FAILURE:
                        self._offset = index2
                        chunk4 = None
                        if self._offset < self._input_size:
                            chunk4 = self._input[self._offset:self._offset + 2]
                        if chunk4 == '==':
                            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                            self._offset = self._offset + 2
                        else:
                            address1 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\'==\'')
                        if address1 is FAILURE:
                            self._offset = index2
                            chunk5 = None
                            if self._offset < self._input_size:
                                chunk5 = self._input[self._offset:self._offset + 1]
                            if chunk5 == '=':
                                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                self._offset = self._offset + 1
                            else:
                                address1 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('\'=\'')
                            if address1 is FAILURE:
                                self._offset = index2
                                chunk6 = None
                                if self._offset < self._input_size:
                                    chunk6 = self._input[self._offset:self._offset + 2]
                                if chunk6 == '!=':
                                    address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                                    self._offset = self._offset + 2
                                else:
                                    address1 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\'!=\'')
                                if address1 is FAILURE:
                                    self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.numeric_comparison(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NumericComparison'][index0] = (address0, self._offset)
        return address0

    def _read_StringCompExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringCompExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_StringValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            address2 = self._read_StringComparison()
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode16(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['StringCompExpr'][index0] = (address0, self._offset)
        return address0

    def _read_StringComparison(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringComparison'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 2]
        if chunk0 == '<=':
            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
            self._offset = self._offset + 2
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'<=\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 1]
            if chunk1 == '<':
                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'<\'')
            if address1 is FAILURE:
                self._offset = index2
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 2]
                if chunk2 == '>=':
                    address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                    self._offset = self._offset + 2
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'>=\'')
                if address1 is FAILURE:
                    self._offset = index2
                    chunk3 = None
                    if self._offset < self._input_size:
                        chunk3 = self._input[self._offset:self._offset + 1]
                    if chunk3 == '>':
                        address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address1 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\'>\'')
                    if address1 is FAILURE:
                        self._offset = index2
                        chunk4 = None
                        if self._offset < self._input_size:
                            chunk4 = self._input[self._offset:self._offset + 2]
                        if chunk4 == '==':
                            address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                            self._offset = self._offset + 2
                        else:
                            address1 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\'==\'')
                        if address1 is FAILURE:
                            self._offset = index2
                            chunk5 = None
                            if self._offset < self._input_size:
                                chunk5 = self._input[self._offset:self._offset + 1]
                            if chunk5 == '=':
                                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                self._offset = self._offset + 1
                            else:
                                address1 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('\'=\'')
                            if address1 is FAILURE:
                                self._offset = index2
                                chunk6 = None
                                if self._offset < self._input_size:
                                    chunk6 = self._input[self._offset:self._offset + 2]
                                if chunk6 == '!=':
                                    address1 = TreeNode(self._input[self._offset:self._offset + 2], self._offset)
                                    self._offset = self._offset + 2
                                else:
                                    address1 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\'!=\'')
                                if address1 is FAILURE:
                                    self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.string_comparison(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StringComparison'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanFunctions(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanFunctions'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_BooleanFunc1()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_BooleanFunc2()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_BooleanSeriesFunc()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['BooleanFunctions'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanFunc1(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanFunc1'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_IsNullFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NotNullFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_StartsWithFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_EndsWithFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_ContainsFunc()
                        if address0 is FAILURE:
                            self._offset = index1
        self._cache['BooleanFunc1'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanFunc2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanFunc2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_IsInFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_SortFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_IfThenElseFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_IfFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_GroupByBooleanFunc()
                        if address0 is FAILURE:
                            self._offset = index1
        self._cache['BooleanFunc2'][index0] = (address0, self._offset)
        return address0

    def _read_IsNullFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IsNullFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'isnull(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'isnull(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_AnyValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.isnull_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IsNullFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NotNullFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NotNullFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 8]
        if chunk0 == 'notnull(':
            address1 = TreeNode(self._input[self._offset:self._offset + 8], self._offset)
            self._offset = self._offset + 8
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'notnull(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_AnyValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.notnull_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NotNullFunc'][index0] = (address0, self._offset)
        return address0

    def _read_StartsWithFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StartsWithFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 11]
        if chunk0 == 'startswith(':
            address1 = TreeNode(self._input[self._offset:self._offset + 11], self._offset)
            self._offset = self._offset + 11
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'startswith(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.starts_with_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StartsWithFunc'][index0] = (address0, self._offset)
        return address0

    def _read_EndsWithFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['EndsWithFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 9]
        if chunk0 == 'endswith(':
            address1 = TreeNode(self._input[self._offset:self._offset + 9], self._offset)
            self._offset = self._offset + 9
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'endswith(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.ends_with_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['EndsWithFunc'][index0] = (address0, self._offset)
        return address0

    def _read_ContainsFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ContainsFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 9]
        if chunk0 == 'contains(':
            address1 = TreeNode(self._input[self._offset:self._offset + 9], self._offset)
            self._offset = self._offset + 9
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'contains(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.contains_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ContainsFunc'][index0] = (address0, self._offset)
        return address0

    def _read_IsInFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IsInFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'isin(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'isin(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericOrStringValueComma()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_ListValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                remaining2, index4, elements3, address10 = 0, self._offset, [], True
                                while address10 is not FAILURE:
                                    address10 = self._read_WS()
                                    if address10 is not FAILURE:
                                        elements3.append(address10)
                                        remaining2 -= 1
                                if remaining2 <= 0:
                                    address9 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                    self._offset = self._offset
                                else:
                                    address9 = FAILURE
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address11 = FAILURE
                                    chunk2 = None
                                    if self._offset < self._input_size:
                                        chunk2 = self._input[self._offset:self._offset + 1]
                                    if chunk2 == ')':
                                        address11 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                        self._offset = self._offset + 1
                                    else:
                                        address11 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append('\')\'')
                                    if address11 is not FAILURE:
                                        elements0.append(address11)
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.is_in_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IsInFunc'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanSeriesFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanSeriesFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_AnyFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_AllFunc()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['BooleanSeriesFunc'][index0] = (address0, self._offset)
        return address0

    def _read_AnyFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'any(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'any(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_BooleanValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.any_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['AnyFunc'][index0] = (address0, self._offset)
        return address0

    def _read_AllFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AllFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'all(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'all(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_BooleanValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.all_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['AllFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NumericValueComma(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericValueComma'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_NumericValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode26(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['NumericValueComma'][index0] = (address0, self._offset)
        return address0

    def _read_NumericValueParen(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericValueParen'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_NumericValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ')':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\')\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode27(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['NumericValueParen'][index0] = (address0, self._offset)
        return address0

    def _read_NumericValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_AddFactor()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_AddExpr()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode28(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['NumericValue'][index0] = (address0, self._offset)
        return address0

    def _read_AddExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AddExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '+':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'+\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 1]
            if chunk1 == '-':
                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'-\'')
            if address1 is FAILURE:
                self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_AddFactor()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.add_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['AddExpr'][index0] = (address0, self._offset)
        return address0

    def _read_AddFactor(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AddFactor'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_NumericFactor()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_MulExpr()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode30(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['AddFactor'][index0] = (address0, self._offset)
        return address0

    def _read_MulExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MulExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '*':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'*\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 1]
            if chunk1 == '/':
                address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'/\'')
            if address1 is FAILURE:
                self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericFactor()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.mul_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MulExpr'][index0] = (address0, self._offset)
        return address0

    def _read_NegExpr(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NegExpr'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '-':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'-\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericFactor()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.neg_expr(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NegExpr'][index0] = (address0, self._offset)
        return address0

    def _read_NumericFactor(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericFactor'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3, elements1 = self._offset, []
        address2 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '(':
            address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address2 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'(\'')
        if address2 is not FAILURE:
            elements1.append(address2)
            address3 = FAILURE
            remaining0, index4, elements2, address4 = 0, self._offset, [], True
            while address4 is not FAILURE:
                address4 = self._read_WS()
                if address4 is not FAILURE:
                    elements2.append(address4)
                    remaining0 -= 1
            if remaining0 <= 0:
                address3 = TreeNode(self._input[index4:self._offset], index4, elements2)
                self._offset = self._offset
            else:
                address3 = FAILURE
            if address3 is not FAILURE:
                elements1.append(address3)
                address5 = FAILURE
                address5 = self._read_NumericValue()
                if address5 is not FAILURE:
                    elements1.append(address5)
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address6 is not FAILURE:
                        elements1.append(address6)
                    else:
                        elements1 = None
                        self._offset = index3
                else:
                    elements1 = None
                    self._offset = index3
            else:
                elements1 = None
                self._offset = index3
        else:
            elements1 = None
            self._offset = index3
        if elements1 is None:
            address1 = FAILURE
        else:
            address1 = TreeNode33(self._input[index3:self._offset], index3, elements1)
            self._offset = self._offset
        if address1 is FAILURE:
            self._offset = index2
            address1 = self._read_NumericFunctions()
            if address1 is FAILURE:
                self._offset = index2
                address1 = self._read_NumericFactor2()
                if address1 is FAILURE:
                    self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address7 = FAILURE
            remaining1, index5, elements3, address8 = 0, self._offset, [], True
            while address8 is not FAILURE:
                address8 = self._read_WS()
                if address8 is not FAILURE:
                    elements3.append(address8)
                    remaining1 -= 1
            if remaining1 <= 0:
                address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                self._offset = self._offset
            else:
                address7 = FAILURE
            if address7 is not FAILURE:
                elements0.append(address7)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['NumericFactor'][index0] = (address0, self._offset)
        return address0

    def _read_NumericFactor2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericFactor2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_NumberConst()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NanConst()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_Variable()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_NegExpr()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['NumericFactor2'][index0] = (address0, self._offset)
        return address0

    def _read_NumericFunctions(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericFunctions'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_NumericFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NumericFunc2()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_NumericSeriesFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_IfThenElseFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_IfFunc()
                        if address0 is FAILURE:
                            self._offset = index1
                            address0 = self._read_GroupByNumericFunc()
                            if address0 is FAILURE:
                                self._offset = index1
        self._cache['NumericFunctions'][index0] = (address0, self._offset)
        return address0

    def _read_NumericFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_AbsFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_PowFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_RoundFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_LengthFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_FindFunc()
                        if address0 is FAILURE:
                            self._offset = index1
                            address0 = self._read_IntFunc()
                            if address0 is FAILURE:
                                self._offset = index1
                                address0 = self._read_NumFillnaFunc()
                                if address0 is FAILURE:
                                    self._offset = index1
        self._cache['NumericFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NumericFunc2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericFunc2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_RowIdFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_MaxListFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_MinListFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_SortFunc()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['NumericFunc2'][index0] = (address0, self._offset)
        return address0

    def _read_AbsFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AbsFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'abs(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'abs(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.abs_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['AbsFunc'][index0] = (address0, self._offset)
        return address0

    def _read_PowFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['PowFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'pow(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'pow(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_NumericValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.pow_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['PowFunc'][index0] = (address0, self._offset)
        return address0

    def _read_RoundFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['RoundFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'round(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'round(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_IntegerConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.round_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['RoundFunc'][index0] = (address0, self._offset)
        return address0

    def _read_LengthFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['LengthFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'length(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'length(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.length_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['LengthFunc'][index0] = (address0, self._offset)
        return address0

    def _read_FindFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['FindFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'find(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'find(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.find_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['FindFunc'][index0] = (address0, self._offset)
        return address0

    def _read_IntFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IntFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'int(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'int(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericOrStringValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.int_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IntFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NumFillnaFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumFillnaFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'fillna(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'fillna(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_NumericValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.fillna_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NumFillnaFunc'][index0] = (address0, self._offset)
        return address0

    def _read_RowIdFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['RowIdFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'rowid()':
            address0 = self._actions.rowid_func(self._input, self._offset, self._offset + 7)
            self._offset = self._offset + 7
        else:
            address0 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'rowid()\'')
        self._cache['RowIdFunc'][index0] = (address0, self._offset)
        return address0

    def _read_MaxListFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MaxListFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'max(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'max(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index3, elements2, address5 = 1, self._offset, [], True
                while address5 is not FAILURE:
                    address5 = self._read_NextNumericValue()
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index3:self._offset], index3, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address6 = FAILURE
                    address6 = self._read_NumericValueParen()
                    if address6 is not FAILURE:
                        elements0.append(address6)
                        address7 = FAILURE
                        chunk1 = None
                        if self._offset < self._input_size:
                            chunk1 = self._input[self._offset:self._offset + 1]
                        if chunk1 == ')':
                            address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address7 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address7 is not FAILURE:
                            elements0.append(address7)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.max_list_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MaxListFunc'][index0] = (address0, self._offset)
        return address0

    def _read_MinListFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MinListFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'min(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'min(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index3, elements2, address5 = 1, self._offset, [], True
                while address5 is not FAILURE:
                    address5 = self._read_NextNumericValue()
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index3:self._offset], index3, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address6 = FAILURE
                    address6 = self._read_NumericValueParen()
                    if address6 is not FAILURE:
                        elements0.append(address6)
                        address7 = FAILURE
                        chunk1 = None
                        if self._offset < self._input_size:
                            chunk1 = self._input[self._offset:self._offset + 1]
                        if chunk1 == ')':
                            address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address7 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address7 is not FAILURE:
                            elements0.append(address7)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.min_list_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MinListFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NumericSeriesFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericSeriesFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_MeanFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_MedianFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_Stdfunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_Semfunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_NumericSeriesFunc2()
                        if address0 is FAILURE:
                            self._offset = index1
        self._cache['NumericSeriesFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NumericSeriesFunc2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericSeriesFunc2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_MinFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_MaxFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_SumFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_CountFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_SizeFunc()
                        if address0 is FAILURE:
                            self._offset = index1
                            address0 = self._read_UniqueFunc()
                            if address0 is FAILURE:
                                self._offset = index1
        self._cache['NumericSeriesFunc2'][index0] = (address0, self._offset)
        return address0

    def _read_MeanFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MeanFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'mean(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'mean(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.mean_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MeanFunc'][index0] = (address0, self._offset)
        return address0

    def _read_MedianFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MedianFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'median(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'median(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.median_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MedianFunc'][index0] = (address0, self._offset)
        return address0

    def _read_Stdfunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Stdfunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'std(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'std(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.std_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['Stdfunc'][index0] = (address0, self._offset)
        return address0

    def _read_Semfunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Semfunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'sem(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'sem(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.sem_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['Semfunc'][index0] = (address0, self._offset)
        return address0

    def _read_MinFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MinFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'min(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'min(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.min_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MinFunc'][index0] = (address0, self._offset)
        return address0

    def _read_MaxFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['MaxFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'max(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'max(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.max_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['MaxFunc'][index0] = (address0, self._offset)
        return address0

    def _read_SumFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['SumFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'sum(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'sum(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.sum_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['SumFunc'][index0] = (address0, self._offset)
        return address0

    def _read_CountFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['CountFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'count(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'count(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericOrStringValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.count_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['CountFunc'][index0] = (address0, self._offset)
        return address0

    def _read_SizeFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['SizeFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'size(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'size(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericOrStringValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.size_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['SizeFunc'][index0] = (address0, self._offset)
        return address0

    def _read_UniqueFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['UniqueFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'unique(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'unique(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index3, elements2, address5 = 1, self._offset, [], True
                while address5 is not FAILURE:
                    address5 = self._read_NextNumericStringValue()
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index3:self._offset], index3, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address6 = FAILURE
                    address6 = self._read_NumericValueParen()
                    if address6 is not FAILURE:
                        elements0.append(address6)
                        address7 = FAILURE
                        chunk1 = None
                        if self._offset < self._input_size:
                            chunk1 = self._input[self._offset:self._offset + 1]
                        if chunk1 == ')':
                            address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address7 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address7 is not FAILURE:
                            elements0.append(address7)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.unique_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['UniqueFunc'][index0] = (address0, self._offset)
        return address0

    def _read_NextNumericValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NextNumericValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_NumericValueComma()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index2, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index2:self._offset], index2, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.next_value(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NextNumericValue'][index0] = (address0, self._offset)
        return address0

    def _read_NextNumericStringValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NextNumericStringValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        address1 = self._read_NumericValueComma()
        if address1 is FAILURE:
            self._offset = index2
            address1 = self._read_StringValueComma()
            if address1 is FAILURE:
                self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.next_value(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NextNumericStringValue'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByBooleanFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByBooleanFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupByAnyFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupByAllFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByIsIncreasing()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByIsDecreasing()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_GroupByApplyFunc()
                        if address0 is FAILURE:
                            self._offset = index1
        self._cache['GroupByBooleanFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByAnyFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByAnyFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'any(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'any(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_any_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByAnyFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByAllFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByAllFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'all(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'all(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_all_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByAllFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByIsIncreasing(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByIsIncreasing'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 14]
        if chunk0 == 'is_increasing(':
            address1 = TreeNode(self._input[self._offset:self._offset + 14], self._offset)
            self._offset = self._offset + 14
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'is_increasing(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_SeriesGroupBy()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_is_increasing_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByIsIncreasing'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByIsDecreasing(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByIsDecreasing'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 14]
        if chunk0 == 'is_decreasing(':
            address1 = TreeNode(self._input[self._offset:self._offset + 14], self._offset)
            self._offset = self._offset + 14
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'is_decreasing(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_SeriesGroupBy()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_is_decreasing_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByIsDecreasing'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNumericFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNumericFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupByNumericFunc2()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupByNumericFunc3()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByNumericFunc4()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByNumericFunc5()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['GroupByNumericFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNumericFunc2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNumericFunc2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupByCumCountFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupByCumMaxFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByCumMinFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByCumProdFunc()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['GroupByNumericFunc2'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNumericFunc3(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNumericFunc3'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupByCumSumFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupByNGroupFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByMeanFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByMedianFunc()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['GroupByNumericFunc3'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNumericFunc4(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNumericFunc4'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupByStdfunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupBySemfunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByMinFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByMaxFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_GroupByProdFunc()
                        if address0 is FAILURE:
                            self._offset = index1
        self._cache['GroupByNumericFunc4'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNumericFunc5(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNumericFunc5'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_GroupBySumFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_GroupByCountFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupBySizeFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_GroupByApplyFunc()
                    if address0 is FAILURE:
                        self._offset = index1
        self._cache['GroupByNumericFunc5'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCumCountFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCumCountFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 9]
        if chunk0 == 'cumcount(':
            address1 = TreeNode(self._input[self._offset:self._offset + 9], self._offset)
            self._offset = self._offset + 9
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'cumcount(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_cum_count_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCumCountFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCumMaxFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCumMaxFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'cummax(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'cummax(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_cum_max_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCumMaxFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCumMinFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCumMinFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'cummin(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'cummin(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_cum_min_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCumMinFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCumProdFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCumProdFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 8]
        if chunk0 == 'cumprod(':
            address1 = TreeNode(self._input[self._offset:self._offset + 8], self._offset)
            self._offset = self._offset + 8
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'cumprod(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_cum_prod_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCumProdFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCumSumFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCumSumFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'cumsum(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'cumsum(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_cum_sum_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCumSumFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByNGroupFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByNGroupFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'ngroup(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'ngroup(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index3 = self._offset
                address4 = self._read_DataFrameGroupBy()
                if address4 is FAILURE:
                    self._offset = index3
                    address4 = self._read_SeriesGroupBy()
                    if address4 is FAILURE:
                        self._offset = index3
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_num_group_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByNGroupFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByMeanFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByMeanFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'mean(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'mean(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_mean_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByMeanFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByMedianFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByMedianFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'median(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'median(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_median_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByMedianFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByStdfunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByStdfunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'std(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'std(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_std_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByStdfunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupBySemfunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupBySemfunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'sem(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'sem(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_sem_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupBySemfunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByMinFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByMinFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'min(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'min(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_min_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByMinFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByMaxFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByMaxFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'max(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'max(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_max_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByMaxFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByProdFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByProdFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'prod(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'prod(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index3, elements2, address5 = 0, self._offset, [], True
                while address5 is not FAILURE:
                    address5 = self._read_GroupBySeriesParam()
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index3:self._offset], index3, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address6 is not FAILURE:
                        elements0.append(address6)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_prod_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByProdFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupBySumFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupBySumFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'sum(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'sum(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_sum_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupBySumFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByCountFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByCountFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'count(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'count(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_GroupBySeriesParam()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_count_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByCountFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupBySizeFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupBySizeFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'size(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'size(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index3 = self._offset
                address4 = self._read_DataFrameGroupBy()
                if address4 is FAILURE:
                    self._offset = index3
                    address4 = self._read_SeriesGroupBy()
                    if address4 is FAILURE:
                        self._offset = index3
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_size_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupBySizeFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupByApplyFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupByApplyFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'apply(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'apply(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_DataFrameGroupBy()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_ExpressionParam()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_apply_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GroupByApplyFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GroupBySeriesParam(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GroupBySeriesParam'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_DataFrameGroupBy()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address5 = FAILURE
                    address5 = self._read_VariableName()
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index4, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index4:self._offset], index4, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                        else:
                            elements0 = None
                            self._offset = index2
                    else:
                        elements0 = None
                        self._offset = index2
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode72(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_SeriesGroupBy()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['GroupBySeriesParam'][index0] = (address0, self._offset)
        return address0

    def _read_DataFrameGroupBy(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['DataFrameGroupBy'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 8]
        if chunk0 == 'groupby(':
            address1 = TreeNode(self._input[self._offset:self._offset + 8], self._offset)
            self._offset = self._offset + 8
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'groupby(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index3 = self._offset
                address4 = self._read_VariableNameList()
                if address4 is FAILURE:
                    self._offset = index3
                    address4 = self._read_AnyValueParen()
                    if address4 is FAILURE:
                        self._offset = index3
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index4, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index4:self._offset], index4, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.groupby_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['DataFrameGroupBy'][index0] = (address0, self._offset)
        return address0

    def _read_SeriesGroupBy(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['SeriesGroupBy'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 15]
        if chunk0 == 'series_groupby(':
            address1 = TreeNode(self._input[self._offset:self._offset + 15], self._offset)
            self._offset = self._offset + 15
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'series_groupby(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_AnyValueComma()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            chunk2 = None
                            if self._offset < self._input_size:
                                chunk2 = self._input[self._offset:self._offset + 3]
                            if chunk2 == 'by=':
                                address8 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
                                self._offset = self._offset + 3
                            else:
                                address8 = FAILURE
                                if self._offset > self._failure:
                                    self._failure = self._offset
                                    self._expected = []
                                if self._offset == self._failure:
                                    self._expected.append('\'by=\'')
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                remaining2, index4, elements3, address10 = 0, self._offset, [], True
                                while address10 is not FAILURE:
                                    address10 = self._read_WS()
                                    if address10 is not FAILURE:
                                        elements3.append(address10)
                                        remaining2 -= 1
                                if remaining2 <= 0:
                                    address9 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                    self._offset = self._offset
                                else:
                                    address9 = FAILURE
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address11 = FAILURE
                                    address11 = self._read_AnyValueParen()
                                    if address11 is not FAILURE:
                                        elements0.append(address11)
                                        address12 = FAILURE
                                        chunk3 = None
                                        if self._offset < self._input_size:
                                            chunk3 = self._input[self._offset:self._offset + 1]
                                        if chunk3 == ')':
                                            address12 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                            self._offset = self._offset + 1
                                        else:
                                            address12 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append('\')\'')
                                        if address12 is not FAILURE:
                                            elements0.append(address12)
                                            address13 = FAILURE
                                            remaining3, index5, elements4, address14 = 0, self._offset, [], True
                                            while address14 is not FAILURE:
                                                address14 = self._read_WS()
                                                if address14 is not FAILURE:
                                                    elements4.append(address14)
                                                    remaining3 -= 1
                                            if remaining3 <= 0:
                                                address13 = TreeNode(self._input[index5:self._offset], index5, elements4)
                                                self._offset = self._offset
                                            else:
                                                address13 = FAILURE
                                            if address13 is not FAILURE:
                                                elements0.append(address13)
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.series_groupby_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['SeriesGroupBy'][index0] = (address0, self._offset)
        return address0

    def _read_ExpressionParam(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ExpressionParam'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'expr=':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'expr=\'')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 13]
            if chunk1 == 'boolean_expr=':
                address1 = TreeNode(self._input[self._offset:self._offset + 13], self._offset)
                self._offset = self._offset + 13
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'boolean_expr=\'')
            if address1 is FAILURE:
                self._offset = index2
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 13]
                if chunk2 == 'numeric_expr=':
                    address1 = TreeNode(self._input[self._offset:self._offset + 13], self._offset)
                    self._offset = self._offset + 13
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'numeric_expr=\'')
                if address1 is FAILURE:
                    self._offset = index2
                    chunk3 = None
                    if self._offset < self._input_size:
                        chunk3 = self._input[self._offset:self._offset + 12]
                    if chunk3 == 'string_expr=':
                        address1 = TreeNode(self._input[self._offset:self._offset + 12], self._offset)
                        self._offset = self._offset + 12
                    else:
                        address1 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\'string_expr=\'')
                    if address1 is FAILURE:
                        self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_ExpressionConst()
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode74(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['ExpressionParam'][index0] = (address0, self._offset)
        return address0

    def _read_StringValueComma(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringValueComma'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_StringValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode75(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['StringValueComma'][index0] = (address0, self._offset)
        return address0

    def _read_StringValueParen(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringValueParen'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_StringValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ')':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\')\'')
            self._offset = index2
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode76(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['StringValueParen'][index0] = (address0, self._offset)
        return address0

    def _read_StringValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        index3, elements1 = self._offset, []
        address2 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '(':
            address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address2 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'(\'')
        if address2 is not FAILURE:
            elements1.append(address2)
            address3 = FAILURE
            remaining0, index4, elements2, address4 = 0, self._offset, [], True
            while address4 is not FAILURE:
                address4 = self._read_WS()
                if address4 is not FAILURE:
                    elements2.append(address4)
                    remaining0 -= 1
            if remaining0 <= 0:
                address3 = TreeNode(self._input[index4:self._offset], index4, elements2)
                self._offset = self._offset
            else:
                address3 = FAILURE
            if address3 is not FAILURE:
                elements1.append(address3)
                address5 = FAILURE
                address5 = self._read_StringValue()
                if address5 is not FAILURE:
                    elements1.append(address5)
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address6 is not FAILURE:
                        elements1.append(address6)
                    else:
                        elements1 = None
                        self._offset = index3
                else:
                    elements1 = None
                    self._offset = index3
            else:
                elements1 = None
                self._offset = index3
        else:
            elements1 = None
            self._offset = index3
        if elements1 is None:
            address1 = FAILURE
        else:
            address1 = TreeNode77(self._input[index3:self._offset], index3, elements1)
            self._offset = self._offset
        if address1 is FAILURE:
            self._offset = index2
            address1 = self._read_StringFunctions()
            if address1 is FAILURE:
                self._offset = index2
                address1 = self._read_StringConst()
                if address1 is FAILURE:
                    self._offset = index2
                    address1 = self._read_NoneConst()
                    if address1 is FAILURE:
                        self._offset = index2
                        address1 = self._read_Variable()
                        if address1 is FAILURE:
                            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address7 = FAILURE
            remaining1, index5, elements3, address8 = 0, self._offset, [], True
            while address8 is not FAILURE:
                address8 = self._read_WS()
                if address8 is not FAILURE:
                    elements3.append(address8)
                    remaining1 -= 1
            if remaining1 <= 0:
                address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                self._offset = self._offset
            else:
                address7 = FAILURE
            if address7 is not FAILURE:
                elements0.append(address7)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['StringValue'][index0] = (address0, self._offset)
        return address0

    def _read_StringFunctions(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringFunctions'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_StringFunc1()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_StringFunc2()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_StringFunc3()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['StringFunctions'][index0] = (address0, self._offset)
        return address0

    def _read_StringFunc1(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringFunc1'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_ConcatFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_PadLeftFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_PadRightFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_StripFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_LeftStripFunc()
                        if address0 is FAILURE:
                            self._offset = index1
                            address0 = self._read_RightStripFunc()
                            if address0 is FAILURE:
                                self._offset = index1
        self._cache['StringFunc1'][index0] = (address0, self._offset)
        return address0

    def _read_StringFunc2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringFunc2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_ReplaceFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_SliceFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GetFunc()
                if address0 is FAILURE:
                    self._offset = index1
                    address0 = self._read_StrFunc()
                    if address0 is FAILURE:
                        self._offset = index1
                        address0 = self._read_StrFillna()
                        if address0 is FAILURE:
                            self._offset = index1
                            address0 = self._read_SortFunc()
                            if address0 is FAILURE:
                                self._offset = index1
        self._cache['StringFunc2'][index0] = (address0, self._offset)
        return address0

    def _read_StringFunc3(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringFunc3'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_IfThenElseFunc()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_IfFunc()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_GroupByApplyFunc()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['StringFunc3'][index0] = (address0, self._offset)
        return address0

    def _read_ConcatFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ConcatFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'concat(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'concat(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.concat_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ConcatFunc'][index0] = (address0, self._offset)
        return address0

    def _read_PadLeftFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['PadLeftFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 8]
        if chunk0 == 'padleft(':
            address1 = TreeNode(self._input[self._offset:self._offset + 8], self._offset)
            self._offset = self._offset + 8
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'padleft(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_NumericValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ',':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\',\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address10 = FAILURE
                                    remaining2, index4, elements3, address11 = 0, self._offset, [], True
                                    while address11 is not FAILURE:
                                        address11 = self._read_WS()
                                        if address11 is not FAILURE:
                                            elements3.append(address11)
                                            remaining2 -= 1
                                    if remaining2 <= 0:
                                        address10 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                        self._offset = self._offset
                                    else:
                                        address10 = FAILURE
                                    if address10 is not FAILURE:
                                        elements0.append(address10)
                                        address12 = FAILURE
                                        address12 = self._read_StringValue()
                                        if address12 is not FAILURE:
                                            elements0.append(address12)
                                            address13 = FAILURE
                                            chunk3 = None
                                            if self._offset < self._input_size:
                                                chunk3 = self._input[self._offset:self._offset + 1]
                                            if chunk3 == ')':
                                                address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                                self._offset = self._offset + 1
                                            else:
                                                address13 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('\')\'')
                                            if address13 is not FAILURE:
                                                elements0.append(address13)
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.pad_left_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['PadLeftFunc'][index0] = (address0, self._offset)
        return address0

    def _read_PadRightFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['PadRightFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 9]
        if chunk0 == 'padright(':
            address1 = TreeNode(self._input[self._offset:self._offset + 9], self._offset)
            self._offset = self._offset + 9
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'padright(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_NumericValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ',':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\',\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address10 = FAILURE
                                    remaining2, index4, elements3, address11 = 0, self._offset, [], True
                                    while address11 is not FAILURE:
                                        address11 = self._read_WS()
                                        if address11 is not FAILURE:
                                            elements3.append(address11)
                                            remaining2 -= 1
                                    if remaining2 <= 0:
                                        address10 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                        self._offset = self._offset
                                    else:
                                        address10 = FAILURE
                                    if address10 is not FAILURE:
                                        elements0.append(address10)
                                        address12 = FAILURE
                                        address12 = self._read_StringValue()
                                        if address12 is not FAILURE:
                                            elements0.append(address12)
                                            address13 = FAILURE
                                            chunk3 = None
                                            if self._offset < self._input_size:
                                                chunk3 = self._input[self._offset:self._offset + 1]
                                            if chunk3 == ')':
                                                address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                                self._offset = self._offset + 1
                                            else:
                                                address13 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('\')\'')
                                            if address13 is not FAILURE:
                                                elements0.append(address13)
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.pad_right_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['PadRightFunc'][index0] = (address0, self._offset)
        return address0

    def _read_StripFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StripFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'strip(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'strip(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    index3 = self._offset
                    index4, elements2 = self._offset, []
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address6 is not FAILURE:
                        elements2.append(address6)
                        address7 = FAILURE
                        remaining1, index5, elements3, address8 = 0, self._offset, [], True
                        while address8 is not FAILURE:
                            address8 = self._read_WS()
                            if address8 is not FAILURE:
                                elements3.append(address8)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                            self._offset = self._offset
                        else:
                            address7 = FAILURE
                        if address7 is not FAILURE:
                            elements2.append(address7)
                            address9 = FAILURE
                            address9 = self._read_StringValue()
                            if address9 is not FAILURE:
                                elements2.append(address9)
                            else:
                                elements2 = None
                                self._offset = index4
                        else:
                            elements2 = None
                            self._offset = index4
                    else:
                        elements2 = None
                        self._offset = index4
                    if elements2 is None:
                        address5 = FAILURE
                    else:
                        address5 = TreeNode82(self._input[index4:self._offset], index4, elements2)
                        self._offset = self._offset
                    if address5 is FAILURE:
                        address5 = TreeNode(self._input[index3:index3], index3)
                        self._offset = index3
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address10 = FAILURE
                        chunk2 = None
                        if self._offset < self._input_size:
                            chunk2 = self._input[self._offset:self._offset + 1]
                        if chunk2 == ')':
                            address10 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address10 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address10 is not FAILURE:
                            elements0.append(address10)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.strip_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StripFunc'][index0] = (address0, self._offset)
        return address0

    def _read_LeftStripFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['LeftStripFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'lstrip(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'lstrip(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    index3 = self._offset
                    index4, elements2 = self._offset, []
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address6 is not FAILURE:
                        elements2.append(address6)
                        address7 = FAILURE
                        remaining1, index5, elements3, address8 = 0, self._offset, [], True
                        while address8 is not FAILURE:
                            address8 = self._read_WS()
                            if address8 is not FAILURE:
                                elements3.append(address8)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                            self._offset = self._offset
                        else:
                            address7 = FAILURE
                        if address7 is not FAILURE:
                            elements2.append(address7)
                            address9 = FAILURE
                            address9 = self._read_StringValue()
                            if address9 is not FAILURE:
                                elements2.append(address9)
                            else:
                                elements2 = None
                                self._offset = index4
                        else:
                            elements2 = None
                            self._offset = index4
                    else:
                        elements2 = None
                        self._offset = index4
                    if elements2 is None:
                        address5 = FAILURE
                    else:
                        address5 = TreeNode84(self._input[index4:self._offset], index4, elements2)
                        self._offset = self._offset
                    if address5 is FAILURE:
                        address5 = TreeNode(self._input[index3:index3], index3)
                        self._offset = index3
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address10 = FAILURE
                        chunk2 = None
                        if self._offset < self._input_size:
                            chunk2 = self._input[self._offset:self._offset + 1]
                        if chunk2 == ')':
                            address10 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address10 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address10 is not FAILURE:
                            elements0.append(address10)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.left_strip_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['LeftStripFunc'][index0] = (address0, self._offset)
        return address0

    def _read_RightStripFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['RightStripFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'rstrip(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'rstrip(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    index3 = self._offset
                    index4, elements2 = self._offset, []
                    address6 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address6 is not FAILURE:
                        elements2.append(address6)
                        address7 = FAILURE
                        remaining1, index5, elements3, address8 = 0, self._offset, [], True
                        while address8 is not FAILURE:
                            address8 = self._read_WS()
                            if address8 is not FAILURE:
                                elements3.append(address8)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                            self._offset = self._offset
                        else:
                            address7 = FAILURE
                        if address7 is not FAILURE:
                            elements2.append(address7)
                            address9 = FAILURE
                            address9 = self._read_StringValue()
                            if address9 is not FAILURE:
                                elements2.append(address9)
                            else:
                                elements2 = None
                                self._offset = index4
                        else:
                            elements2 = None
                            self._offset = index4
                    else:
                        elements2 = None
                        self._offset = index4
                    if elements2 is None:
                        address5 = FAILURE
                    else:
                        address5 = TreeNode86(self._input[index4:self._offset], index4, elements2)
                        self._offset = self._offset
                    if address5 is FAILURE:
                        address5 = TreeNode(self._input[index3:index3], index3)
                        self._offset = index3
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address10 = FAILURE
                        chunk2 = None
                        if self._offset < self._input_size:
                            chunk2 = self._input[self._offset:self._offset + 1]
                        if chunk2 == ')':
                            address10 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address10 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\')\'')
                        if address10 is not FAILURE:
                            elements0.append(address10)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.right_strip_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['RightStripFunc'][index0] = (address0, self._offset)
        return address0

    def _read_ReplaceFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ReplaceFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 8]
        if chunk0 == 'replace(':
            address1 = TreeNode(self._input[self._offset:self._offset + 8], self._offset)
            self._offset = self._offset + 8
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'replace(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ',':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\',\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address10 = FAILURE
                                    remaining2, index4, elements3, address11 = 0, self._offset, [], True
                                    while address11 is not FAILURE:
                                        address11 = self._read_WS()
                                        if address11 is not FAILURE:
                                            elements3.append(address11)
                                            remaining2 -= 1
                                    if remaining2 <= 0:
                                        address10 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                        self._offset = self._offset
                                    else:
                                        address10 = FAILURE
                                    if address10 is not FAILURE:
                                        elements0.append(address10)
                                        address12 = FAILURE
                                        address12 = self._read_StringConst()
                                        if address12 is not FAILURE:
                                            elements0.append(address12)
                                            address13 = FAILURE
                                            chunk3 = None
                                            if self._offset < self._input_size:
                                                chunk3 = self._input[self._offset:self._offset + 1]
                                            if chunk3 == ')':
                                                address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                                self._offset = self._offset + 1
                                            else:
                                                address13 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('\')\'')
                                            if address13 is not FAILURE:
                                                elements0.append(address13)
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.replace_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ReplaceFunc'][index0] = (address0, self._offset)
        return address0

    def _read_SliceFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['SliceFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'slice(':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'slice(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_IntegerConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                index4 = self._offset
                                index5, elements3 = self._offset, []
                                address10 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ',':
                                    address10 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address10 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\',\'')
                                if address10 is not FAILURE:
                                    elements3.append(address10)
                                    address11 = FAILURE
                                    remaining2, index6, elements4, address12 = 0, self._offset, [], True
                                    while address12 is not FAILURE:
                                        address12 = self._read_WS()
                                        if address12 is not FAILURE:
                                            elements4.append(address12)
                                            remaining2 -= 1
                                    if remaining2 <= 0:
                                        address11 = TreeNode(self._input[index6:self._offset], index6, elements4)
                                        self._offset = self._offset
                                    else:
                                        address11 = FAILURE
                                    if address11 is not FAILURE:
                                        elements3.append(address11)
                                        address13 = FAILURE
                                        address13 = self._read_IntegerConst()
                                        if address13 is not FAILURE:
                                            elements3.append(address13)
                                        else:
                                            elements3 = None
                                            self._offset = index5
                                    else:
                                        elements3 = None
                                        self._offset = index5
                                else:
                                    elements3 = None
                                    self._offset = index5
                                if elements3 is None:
                                    address9 = FAILURE
                                else:
                                    address9 = TreeNode89(self._input[index5:self._offset], index5, elements3)
                                    self._offset = self._offset
                                if address9 is FAILURE:
                                    address9 = TreeNode(self._input[index4:index4], index4)
                                    self._offset = index4
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address14 = FAILURE
                                    chunk3 = None
                                    if self._offset < self._input_size:
                                        chunk3 = self._input[self._offset:self._offset + 1]
                                    if chunk3 == ')':
                                        address14 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                        self._offset = self._offset + 1
                                    else:
                                        address14 = FAILURE
                                        if self._offset > self._failure:
                                            self._failure = self._offset
                                            self._expected = []
                                        if self._offset == self._failure:
                                            self._expected.append('\')\'')
                                    if address14 is not FAILURE:
                                        elements0.append(address14)
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.slice_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['SliceFunc'][index0] = (address0, self._offset)
        return address0

    def _read_GetFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['GetFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'get(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'get(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_IntegerConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.get_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['GetFunc'][index0] = (address0, self._offset)
        return address0

    def _read_StrFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StrFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'str(':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'str(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_NumericOrStringValueParen()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ')':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\')\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.str_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StrFunc'][index0] = (address0, self._offset)
        return address0

    def _read_StrFillna(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StrFillna'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 7]
        if chunk0 == 'fillna(':
            address1 = TreeNode(self._input[self._offset:self._offset + 7], self._offset)
            self._offset = self._offset + 7
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'fillna(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_StringValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_StringValue()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.fillna_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StrFillna'][index0] = (address0, self._offset)
        return address0

    def _read_NumericOrStringValueComma(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericOrStringValueComma'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_NumericValueComma()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_StringValueComma()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['NumericOrStringValueComma'][index0] = (address0, self._offset)
        return address0

    def _read_NumericOrStringValueParen(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumericOrStringValueParen'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_NumericValueParen()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_StringValueParen()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['NumericOrStringValueParen'][index0] = (address0, self._offset)
        return address0

    def _read_AnyValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_BooleanValue()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NumericValue()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_StringValue()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['AnyValue'][index0] = (address0, self._offset)
        return address0

    def _read_AnyValueElse(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyValueElse'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BooleanValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 4]
            if chunk0 == 'else':
                address2 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
                self._offset = self._offset + 4
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\'else\'')
            self._offset = index3
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode93(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            index4, elements1 = self._offset, []
            address3 = FAILURE
            address3 = self._read_NumericValue()
            if address3 is not FAILURE:
                elements1.append(address3)
                address4 = FAILURE
                index5 = self._offset
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 4]
                if chunk1 == 'else':
                    address4 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
                    self._offset = self._offset + 4
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'else\'')
                self._offset = index5
                if address4 is not FAILURE:
                    address4 = TreeNode(self._input[self._offset:self._offset], self._offset)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements1.append(address4)
                else:
                    elements1 = None
                    self._offset = index4
            else:
                elements1 = None
                self._offset = index4
            if elements1 is None:
                address0 = FAILURE
            else:
                address0 = TreeNode94(self._input[index4:self._offset], index4, elements1)
                self._offset = self._offset
            if address0 is FAILURE:
                self._offset = index1
                index6, elements2 = self._offset, []
                address5 = FAILURE
                address5 = self._read_StringValue()
                if address5 is not FAILURE:
                    elements2.append(address5)
                    address6 = FAILURE
                    index7 = self._offset
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset:self._offset + 4]
                    if chunk2 == 'else':
                        address6 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
                        self._offset = self._offset + 4
                    else:
                        address6 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\'else\'')
                    self._offset = index7
                    if address6 is not FAILURE:
                        address6 = TreeNode(self._input[self._offset:self._offset], self._offset)
                        self._offset = self._offset
                    else:
                        address6 = FAILURE
                    if address6 is not FAILURE:
                        elements2.append(address6)
                    else:
                        elements2 = None
                        self._offset = index6
                else:
                    elements2 = None
                    self._offset = index6
                if elements2 is None:
                    address0 = FAILURE
                else:
                    address0 = TreeNode95(self._input[index6:self._offset], index6, elements2)
                    self._offset = self._offset
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['AnyValueElse'][index0] = (address0, self._offset)
        return address0

    def _read_AnyValueComma(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyValueComma'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BooleanValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ',':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\',\'')
            self._offset = index3
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode96(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NumericValueComma()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_StringValueComma()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['AnyValueComma'][index0] = (address0, self._offset)
        return address0

    def _read_AnyValueParen(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['AnyValueParen'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        index2, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_BooleanValue()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            chunk0 = None
            if self._offset < self._input_size:
                chunk0 = self._input[self._offset:self._offset + 1]
            if chunk0 == ')':
                address2 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                self._offset = self._offset + 1
            else:
                address2 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('\')\'')
            self._offset = index3
            if address2 is not FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index2
        else:
            elements0 = None
            self._offset = index2
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode97(self._input[index2:self._offset], index2, elements0)
            self._offset = self._offset
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_NumericValueParen()
            if address0 is FAILURE:
                self._offset = index1
                address0 = self._read_StringValueParen()
                if address0 is FAILURE:
                    self._offset = index1
        self._cache['AnyValueParen'][index0] = (address0, self._offset)
        return address0

    def _read_ListValue(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ListValue'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '[':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'[\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                index3 = self._offset
                index4, elements2 = self._offset, []
                address5 = FAILURE
                address5 = self._read_AnyValue()
                if address5 is not FAILURE:
                    elements2.append(address5)
                    address6 = FAILURE
                    remaining1, index5, elements3, address7 = 0, self._offset, [], True
                    while address7 is not FAILURE:
                        index6, elements4 = self._offset, []
                        address8 = FAILURE
                        chunk1 = None
                        if self._offset < self._input_size:
                            chunk1 = self._input[self._offset:self._offset + 1]
                        if chunk1 == ',':
                            address8 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address8 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\',\'')
                        if address8 is not FAILURE:
                            elements4.append(address8)
                            address9 = FAILURE
                            remaining2, index7, elements5, address10 = 0, self._offset, [], True
                            while address10 is not FAILURE:
                                address10 = self._read_WS()
                                if address10 is not FAILURE:
                                    elements5.append(address10)
                                    remaining2 -= 1
                            if remaining2 <= 0:
                                address9 = TreeNode(self._input[index7:self._offset], index7, elements5)
                                self._offset = self._offset
                            else:
                                address9 = FAILURE
                            if address9 is not FAILURE:
                                elements4.append(address9)
                                address11 = FAILURE
                                address11 = self._read_AnyValue()
                                if address11 is not FAILURE:
                                    elements4.append(address11)
                                else:
                                    elements4 = None
                                    self._offset = index6
                            else:
                                elements4 = None
                                self._offset = index6
                        else:
                            elements4 = None
                            self._offset = index6
                        if elements4 is None:
                            address7 = FAILURE
                        else:
                            address7 = TreeNode99(self._input[index6:self._offset], index6, elements4)
                            self._offset = self._offset
                        if address7 is not FAILURE:
                            elements3.append(address7)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address6 = TreeNode(self._input[index5:self._offset], index5, elements3)
                        self._offset = self._offset
                    else:
                        address6 = FAILURE
                    if address6 is not FAILURE:
                        elements2.append(address6)
                    else:
                        elements2 = None
                        self._offset = index4
                else:
                    elements2 = None
                    self._offset = index4
                if elements2 is None:
                    address4 = FAILURE
                else:
                    address4 = TreeNode98(self._input[index4:self._offset], index4, elements2)
                    self._offset = self._offset
                if address4 is FAILURE:
                    address4 = TreeNode(self._input[index3:index3], index3)
                    self._offset = index3
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address12 = FAILURE
                    chunk2 = None
                    if self._offset < self._input_size:
                        chunk2 = self._input[self._offset:self._offset + 1]
                    if chunk2 == ']':
                        address12 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address12 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\']\'')
                    if address12 is not FAILURE:
                        elements0.append(address12)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.list_value(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ListValue'][index0] = (address0, self._offset)
        return address0

    def _read_IfThenElseFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IfThenElseFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 11]
        if chunk0 == 'ifthenelse(':
            address1 = TreeNode(self._input[self._offset:self._offset + 11], self._offset)
            self._offset = self._offset + 11
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'ifthenelse(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_BooleanValueComma()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_AnyValueComma()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ',':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\',\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address10 = FAILURE
                                    remaining2, index4, elements3, address11 = 0, self._offset, [], True
                                    while address11 is not FAILURE:
                                        address11 = self._read_WS()
                                        if address11 is not FAILURE:
                                            elements3.append(address11)
                                            remaining2 -= 1
                                    if remaining2 <= 0:
                                        address10 = TreeNode(self._input[index4:self._offset], index4, elements3)
                                        self._offset = self._offset
                                    else:
                                        address10 = FAILURE
                                    if address10 is not FAILURE:
                                        elements0.append(address10)
                                        address12 = FAILURE
                                        address12 = self._read_AnyValueParen()
                                        if address12 is not FAILURE:
                                            elements0.append(address12)
                                            address13 = FAILURE
                                            chunk3 = None
                                            if self._offset < self._input_size:
                                                chunk3 = self._input[self._offset:self._offset + 1]
                                            if chunk3 == ')':
                                                address13 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                                self._offset = self._offset + 1
                                            else:
                                                address13 = FAILURE
                                                if self._offset > self._failure:
                                                    self._failure = self._offset
                                                    self._expected = []
                                                if self._offset == self._failure:
                                                    self._expected.append('\')\'')
                                            if address13 is not FAILURE:
                                                elements0.append(address13)
                                            else:
                                                elements0 = None
                                                self._offset = index1
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.ifthenelse_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IfThenElseFunc'][index0] = (address0, self._offset)
        return address0

    def _read_IfFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IfFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 3]
        if chunk0 == 'if(':
            address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'if(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_BooleanValue()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    address5 = self._read_Then()
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        address6 = self._read_AnyValueElse()
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address7 = FAILURE
                            remaining1, index3, elements2, address8 = 0, self._offset, [], True
                            while address8 is not FAILURE:
                                address8 = self._read_ElseIfFunc()
                                if address8 is not FAILURE:
                                    elements2.append(address8)
                                    remaining1 -= 1
                            if remaining1 <= 0:
                                address7 = TreeNode(self._input[index3:self._offset], index3, elements2)
                                self._offset = self._offset
                            else:
                                address7 = FAILURE
                            if address7 is not FAILURE:
                                elements0.append(address7)
                                address9 = FAILURE
                                address9 = self._read_Else()
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                    address10 = FAILURE
                                    address10 = self._read_AnyValueParen()
                                    if address10 is not FAILURE:
                                        elements0.append(address10)
                                        address11 = FAILURE
                                        chunk1 = None
                                        if self._offset < self._input_size:
                                            chunk1 = self._input[self._offset:self._offset + 1]
                                        if chunk1 == ')':
                                            address11 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                            self._offset = self._offset + 1
                                        else:
                                            address11 = FAILURE
                                            if self._offset > self._failure:
                                                self._failure = self._offset
                                                self._expected = []
                                            if self._offset == self._failure:
                                                self._expected.append('\')\'')
                                        if address11 is not FAILURE:
                                            elements0.append(address11)
                                        else:
                                            elements0 = None
                                            self._offset = index1
                                    else:
                                        elements0 = None
                                        self._offset = index1
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.if_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IfFunc'][index0] = (address0, self._offset)
        return address0

    def _read_ElseIfFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ElseIfFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_Elseif()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            address2 = self._read_BooleanValue()
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                address3 = self._read_Then()
                if address3 is not FAILURE:
                    elements0.append(address3)
                    address4 = FAILURE
                    address4 = self._read_AnyValueElse()
                    if address4 is not FAILURE:
                        elements0.append(address4)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode102(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['ElseIfFunc'][index0] = (address0, self._offset)
        return address0

    def _read_SortFunc(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['SortFunc'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'sort(':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'sort(\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_AnyValueComma()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    chunk1 = None
                    if self._offset < self._input_size:
                        chunk1 = self._input[self._offset:self._offset + 1]
                    if chunk1 == ',':
                        address5 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                        self._offset = self._offset + 1
                    else:
                        address5 = FAILURE
                        if self._offset > self._failure:
                            self._failure = self._offset
                            self._expected = []
                        if self._offset == self._failure:
                            self._expected.append('\',\'')
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address6 = FAILURE
                        remaining1, index3, elements2, address7 = 0, self._offset, [], True
                        while address7 is not FAILURE:
                            address7 = self._read_WS()
                            if address7 is not FAILURE:
                                elements2.append(address7)
                                remaining1 -= 1
                        if remaining1 <= 0:
                            address6 = TreeNode(self._input[index3:self._offset], index3, elements2)
                            self._offset = self._offset
                        else:
                            address6 = FAILURE
                        if address6 is not FAILURE:
                            elements0.append(address6)
                            address8 = FAILURE
                            address8 = self._read_BooleanConst()
                            if address8 is not FAILURE:
                                elements0.append(address8)
                                address9 = FAILURE
                                chunk2 = None
                                if self._offset < self._input_size:
                                    chunk2 = self._input[self._offset:self._offset + 1]
                                if chunk2 == ')':
                                    address9 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                                    self._offset = self._offset + 1
                                else:
                                    address9 = FAILURE
                                    if self._offset > self._failure:
                                        self._failure = self._offset
                                        self._expected = []
                                    if self._offset == self._failure:
                                        self._expected.append('\')\'')
                                if address9 is not FAILURE:
                                    elements0.append(address9)
                                else:
                                    elements0 = None
                                    self._offset = index1
                            else:
                                elements0 = None
                                self._offset = index1
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.sort_func(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['SortFunc'][index0] = (address0, self._offset)
        return address0

    def _read_Then(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Then'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'then':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"then"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['Then'][index0] = (address0, self._offset)
        return address0

    def _read_Else(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Else'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'else':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"else"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['Else'][index0] = (address0, self._offset)
        return address0

    def _read_Elseif(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Elseif'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 6]
        if chunk0 == 'elseif':
            address1 = TreeNode(self._input[self._offset:self._offset + 6], self._offset)
            self._offset = self._offset + 6
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"elseif"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address3 = FAILURE
                remaining0, index3, elements1, address4 = 0, self._offset, [], True
                while address4 is not FAILURE:
                    address4 = self._read_WS()
                    if address4 is not FAILURE:
                        elements1.append(address4)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address3 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address3 = FAILURE
                if address3 is not FAILURE:
                    elements0.append(address3)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = TreeNode(self._input[index1:self._offset], index1, elements0)
            self._offset = self._offset
        self._cache['Elseif'][index0] = (address0, self._offset)
        return address0

    def _read_BooleanConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['BooleanConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_TrueConst()
        if address0 is FAILURE:
            self._offset = index1
            index2, elements0 = self._offset, []
            address1 = FAILURE
            address1 = self._read_FalseConst()
            if address1 is not FAILURE:
                elements0.append(address1)
                address2 = FAILURE
                remaining0, index3, elements1, address3 = 0, self._offset, [], True
                while address3 is not FAILURE:
                    address3 = self._read_WS()
                    if address3 is not FAILURE:
                        elements1.append(address3)
                        remaining0 -= 1
                if remaining0 <= 0:
                    address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                    self._offset = self._offset
                else:
                    address2 = FAILURE
                if address2 is not FAILURE:
                    elements0.append(address2)
                else:
                    elements0 = None
                    self._offset = index2
            else:
                elements0 = None
                self._offset = index2
            if elements0 is None:
                address0 = FAILURE
            else:
                address0 = TreeNode104(self._input[index2:self._offset], index2, elements0)
                self._offset = self._offset
            if address0 is FAILURE:
                self._offset = index1
        self._cache['BooleanConst'][index0] = (address0, self._offset)
        return address0

    def _read_TrueConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['TrueConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'true':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"true"')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 4]
            if chunk1 == 'True':
                address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
                self._offset = self._offset + 4
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"True"')
            if address1 is FAILURE:
                self._offset = index2
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 4]
                if chunk2 == 'TRUE':
                    address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
                    self._offset = self._offset + 4
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"TRUE"')
                if address1 is FAILURE:
                    self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            address2 = self._read_ANC()
            self._offset = index3
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.true_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['TrueConst'][index0] = (address0, self._offset)
        return address0

    def _read_FalseConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['FalseConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 5]
        if chunk0 == 'false':
            address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
            self._offset = self._offset + 5
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"false"')
        if address1 is FAILURE:
            self._offset = index2
            chunk1 = None
            if self._offset < self._input_size:
                chunk1 = self._input[self._offset:self._offset + 5]
            if chunk1 == 'False':
                address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
                self._offset = self._offset + 5
            else:
                address1 = FAILURE
                if self._offset > self._failure:
                    self._failure = self._offset
                    self._expected = []
                if self._offset == self._failure:
                    self._expected.append('"False"')
            if address1 is FAILURE:
                self._offset = index2
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 5]
                if chunk2 == 'FALSE':
                    address1 = TreeNode(self._input[self._offset:self._offset + 5], self._offset)
                    self._offset = self._offset + 5
                else:
                    address1 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"FALSE"')
                if address1 is FAILURE:
                    self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index3 = self._offset
            address2 = self._read_ANC()
            self._offset = index3
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.false_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['FalseConst'][index0] = (address0, self._offset)
        return address0

    def _read_NoneConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NoneConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 4]
        if chunk0 == 'None':
            address1 = TreeNode(self._input[self._offset:self._offset + 4], self._offset)
            self._offset = self._offset + 4
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"None"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.none_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NoneConst'][index0] = (address0, self._offset)
        return address0

    def _read_NanConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NanConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 3]
        if chunk0 == 'Nan':
            address1 = TreeNode(self._input[self._offset:self._offset + 3], self._offset)
            self._offset = self._offset + 3
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"Nan"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            index2 = self._offset
            address2 = self._read_ANC()
            self._offset = index2
            if address2 is FAILURE:
                address2 = TreeNode(self._input[self._offset:self._offset], self._offset)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.nan_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['NanConst'][index0] = (address0, self._offset)
        return address0

    def _read_NumberConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['NumberConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_FloatConst()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_IntegerConst()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['NumberConst'][index0] = (address0, self._offset)
        return address0

    def _read_IntegerConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['IntegerConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '-':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'-\'')
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 1, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_1.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[0-9]')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                remaining1, index4, elements2, address5 = 0, self._offset, [], True
                while address5 is not FAILURE:
                    address5 = self._read_WS()
                    if address5 is not FAILURE:
                        elements2.append(address5)
                        remaining1 -= 1
                if remaining1 <= 0:
                    address4 = TreeNode(self._input[index4:self._offset], index4, elements2)
                    self._offset = self._offset
                else:
                    address4 = FAILURE
                if address4 is not FAILURE:
                    elements0.append(address4)
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.integer_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['IntegerConst'][index0] = (address0, self._offset)
        return address0

    def _read_FloatConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['FloatConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        index2 = self._offset
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '-':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'-\'')
        if address1 is FAILURE:
            address1 = TreeNode(self._input[index2:index2], index2)
            self._offset = index2
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index3, elements1, address3 = 1, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_2.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[0-9]')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index3:self._offset], index3, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 1]
                if chunk2 == '.':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'.\'')
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index4, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        chunk3 = None
                        if self._offset < self._input_size:
                            chunk3 = self._input[self._offset:self._offset + 1]
                        if chunk3 is not None and Grammar.REGEX_3.search(chunk3):
                            address6 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address6 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('[0-9]')
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index4:self._offset], index4, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address7 = FAILURE
                        remaining2, index5, elements3, address8 = 0, self._offset, [], True
                        while address8 is not FAILURE:
                            address8 = self._read_WS()
                            if address8 is not FAILURE:
                                elements3.append(address8)
                                remaining2 -= 1
                        if remaining2 <= 0:
                            address7 = TreeNode(self._input[index5:self._offset], index5, elements3)
                            self._offset = self._offset
                        else:
                            address7 = FAILURE
                        if address7 is not FAILURE:
                            elements0.append(address7)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.float_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['FloatConst'][index0] = (address0, self._offset)
        return address0

    def _read_ExpressionConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ExpressionConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        address1 = self._read_StringConst()
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.expression_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['ExpressionConst'][index0] = (address0, self._offset)
        return address0

    def _read_StringConst(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringConst'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_StringConst1()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_StringConst2()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['StringConst'][index0] = (address0, self._offset)
        return address0

    def _read_StringConst1(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringConst1'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '\'':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"\'"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_4.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[^\']')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 1]
                if chunk2 == '\'':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"\'"')
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index3, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        address6 = self._read_WS()
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index3:self._offset], index3, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.string_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StringConst1'][index0] = (address0, self._offset)
        return address0

    def _read_StringConst2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['StringConst2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '"':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'"\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_5.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[^"]')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 1]
                if chunk2 == '"':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'"\'')
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index3, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        address6 = self._read_WS()
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index3:self._offset], index3, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.string_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['StringConst2'][index0] = (address0, self._offset)
        return address0

    def _read_ANC(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['ANC'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_6.search(chunk0):
            address0 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address0 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('[a-zA-Z0-9_]')
        self._cache['ANC'][index0] = (address0, self._offset)
        return address0

    def _read_WS(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['WS'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_7.search(chunk0):
            address0 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address0 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('[ ]')
        self._cache['WS'][index0] = (address0, self._offset)
        return address0

    def _read_EndOfInput(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['EndOfInput'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        if self._offset < self._input_size:
            address0 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address0 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('<any char>')
        self._offset = index1
        if address0 is FAILURE:
            address0 = TreeNode(self._input[self._offset:self._offset], self._offset)
            self._offset = self._offset
        else:
            address0 = FAILURE
        self._cache['EndOfInput'][index0] = (address0, self._offset)
        return address0

    def _read_VariableName(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['VariableName'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1 = self._offset
        address0 = self._read_VariableConst1()
        if address0 is FAILURE:
            self._offset = index1
            address0 = self._read_VariableConst2()
            if address0 is FAILURE:
                self._offset = index1
        self._cache['VariableName'][index0] = (address0, self._offset)
        return address0

    def _read_VariableConst1(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['VariableConst1'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '\'':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('"\'"')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_8.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[^\']')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 1]
                if chunk2 == '\'':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('"\'"')
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index3, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        address6 = self._read_WS()
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index3:self._offset], index3, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.variable_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['VariableConst1'][index0] = (address0, self._offset)
        return address0

    def _read_VariableConst2(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['VariableConst2'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '"':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'"\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                chunk1 = None
                if self._offset < self._input_size:
                    chunk1 = self._input[self._offset:self._offset + 1]
                if chunk1 is not None and Grammar.REGEX_9.search(chunk1):
                    address3 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address3 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('[^"]')
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                chunk2 = None
                if self._offset < self._input_size:
                    chunk2 = self._input[self._offset:self._offset + 1]
                if chunk2 == '"':
                    address4 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                    self._offset = self._offset + 1
                else:
                    address4 = FAILURE
                    if self._offset > self._failure:
                        self._failure = self._offset
                        self._expected = []
                    if self._offset == self._failure:
                        self._expected.append('\'"\'')
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index3, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        address6 = self._read_WS()
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index3:self._offset], index3, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.variable_const(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['VariableConst2'][index0] = (address0, self._offset)
        return address0

    def _read_Variable(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['Variable'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 is not None and Grammar.REGEX_10.search(chunk0):
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('[a-zA-Z_]')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_ANC()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.variable(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['Variable'][index0] = (address0, self._offset)
        return address0

    def _read_VariableNameList(self):
        address0, index0 = FAILURE, self._offset
        cached = self._cache['VariableNameList'].get(index0)
        if cached:
            self._offset = cached[1]
            return cached[0]
        index1, elements0 = self._offset, []
        address1 = FAILURE
        chunk0 = None
        if self._offset < self._input_size:
            chunk0 = self._input[self._offset:self._offset + 1]
        if chunk0 == '[':
            address1 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
            self._offset = self._offset + 1
        else:
            address1 = FAILURE
            if self._offset > self._failure:
                self._failure = self._offset
                self._expected = []
            if self._offset == self._failure:
                self._expected.append('\'[\'')
        if address1 is not FAILURE:
            elements0.append(address1)
            address2 = FAILURE
            remaining0, index2, elements1, address3 = 0, self._offset, [], True
            while address3 is not FAILURE:
                address3 = self._read_WS()
                if address3 is not FAILURE:
                    elements1.append(address3)
                    remaining0 -= 1
            if remaining0 <= 0:
                address2 = TreeNode(self._input[index2:self._offset], index2, elements1)
                self._offset = self._offset
            else:
                address2 = FAILURE
            if address2 is not FAILURE:
                elements0.append(address2)
                address4 = FAILURE
                address4 = self._read_VariableName()
                if address4 is not FAILURE:
                    elements0.append(address4)
                    address5 = FAILURE
                    remaining1, index3, elements2, address6 = 0, self._offset, [], True
                    while address6 is not FAILURE:
                        index4, elements3 = self._offset, []
                        address7 = FAILURE
                        chunk1 = None
                        if self._offset < self._input_size:
                            chunk1 = self._input[self._offset:self._offset + 1]
                        if chunk1 == ',':
                            address7 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address7 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\',\'')
                        if address7 is not FAILURE:
                            elements3.append(address7)
                            address8 = FAILURE
                            remaining2, index5, elements4, address9 = 0, self._offset, [], True
                            while address9 is not FAILURE:
                                address9 = self._read_WS()
                                if address9 is not FAILURE:
                                    elements4.append(address9)
                                    remaining2 -= 1
                            if remaining2 <= 0:
                                address8 = TreeNode(self._input[index5:self._offset], index5, elements4)
                                self._offset = self._offset
                            else:
                                address8 = FAILURE
                            if address8 is not FAILURE:
                                elements3.append(address8)
                                address10 = FAILURE
                                address10 = self._read_VariableName()
                                if address10 is not FAILURE:
                                    elements3.append(address10)
                                else:
                                    elements3 = None
                                    self._offset = index4
                            else:
                                elements3 = None
                                self._offset = index4
                        else:
                            elements3 = None
                            self._offset = index4
                        if elements3 is None:
                            address6 = FAILURE
                        else:
                            address6 = TreeNode107(self._input[index4:self._offset], index4, elements3)
                            self._offset = self._offset
                        if address6 is not FAILURE:
                            elements2.append(address6)
                            remaining1 -= 1
                    if remaining1 <= 0:
                        address5 = TreeNode(self._input[index3:self._offset], index3, elements2)
                        self._offset = self._offset
                    else:
                        address5 = FAILURE
                    if address5 is not FAILURE:
                        elements0.append(address5)
                        address11 = FAILURE
                        chunk2 = None
                        if self._offset < self._input_size:
                            chunk2 = self._input[self._offset:self._offset + 1]
                        if chunk2 == ']':
                            address11 = TreeNode(self._input[self._offset:self._offset + 1], self._offset)
                            self._offset = self._offset + 1
                        else:
                            address11 = FAILURE
                            if self._offset > self._failure:
                                self._failure = self._offset
                                self._expected = []
                            if self._offset == self._failure:
                                self._expected.append('\']\'')
                        if address11 is not FAILURE:
                            elements0.append(address11)
                        else:
                            elements0 = None
                            self._offset = index1
                    else:
                        elements0 = None
                        self._offset = index1
                else:
                    elements0 = None
                    self._offset = index1
            else:
                elements0 = None
                self._offset = index1
        else:
            elements0 = None
            self._offset = index1
        if elements0 is None:
            address0 = FAILURE
        else:
            address0 = self._actions.variable_name_list(self._input, index1, self._offset, elements0)
            self._offset = self._offset
        self._cache['VariableNameList'][index0] = (address0, self._offset)
        return address0


class Parser(Grammar):
    def __init__(self, input, actions, types):
        self._input = input
        self._input_size = len(input)
        self._actions = actions
        self._types = types
        self._offset = 0
        self._cache = defaultdict(dict)
        self._failure = 0
        self._expected = []

    def parse(self):
        tree = self._read_ChecksExpression()
        if tree is not FAILURE and self._offset == self._input_size:
            return tree
        if not self._expected:
            self._failure = self._offset
            self._expected.append('<EOF>')
        raise ParseError(format_error(self._input, self._failure, self._expected))


def format_error(input, offset, expected):
    lines, line_no, position = input.split('\n'), 0, 0
    while position <= offset:
        position += len(lines[line_no]) + 1
        line_no += 1
    message, line = 'Line ' + str(line_no) + ': expected ' + ', '.join(expected) + '\n', lines[line_no - 1]
    message += line + '\n'
    position -= len(line) + 1
    message += ' ' * (offset - position)
    return message + '^'

def parse(input, actions=None, types=None):
    parser = Parser(input, actions, types)
    return parser.parse()
