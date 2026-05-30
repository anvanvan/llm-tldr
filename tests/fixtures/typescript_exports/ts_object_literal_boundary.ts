// Fixture: object-literal arrow — boundary guard test
// Tests: _get_variable_declarator_name boundary stops at "pair"
//
// With "export const", the arrow inside the object property is encountered
// by _extract_ts_function:
//   - _get_variable_declarator_name is invoked, stops at "pair" boundary
//     → returns None for the arrow, so it does NOT get name "handler"
//   - _get_pair_property_name provides "onClick" as the name

export const handler = {
    onClick: (): void => {
        console.log("clicked");
    },
};
