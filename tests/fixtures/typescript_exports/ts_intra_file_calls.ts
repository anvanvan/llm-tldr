// Fixture: intra-file arrow-in-const call detection
// Tests: _collect_ts_definitions inline lexical_declaration handler

const inner = (): number => {
    return 42;
};

const outer = (): number => {
    return inner();
};
