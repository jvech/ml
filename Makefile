include config.mk

CC 		= clang
CFLAGS 	= -std=gnu11 -Wall -g
BIN 	= ml
OBJDIR 	= objs
SRC 	= $(wildcard src/*.c)
HEADERS = $(wildcard src/*.h)
OBJS 	= $(SRC:src/%.c=${OBJDIR}/%.o) 
DLIBS 	= -lm $(shell pkg-config --libs-only-l blas json-c)
.PHONY: clean all run

all: build

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir ${OBJDIR}

$(OBJDIR)/%.o: src/%.c $(HEADERS)
	${CC} -c -o $@ $< ${CFLAGS}

build: $(OBJS)
	${CC} ${DLIBS} -o ${BIN} ${OBJS}

install: all
	@# binary
	install -d $(BINPREFIX)
	install -m 755 ${BIN} $(BINPREFIX)/${BIN}
	@#man page
	install -d $(MANPREFIX)/man1
	install -m 644 doc/ml.1 $(MANPREFIX)/man1/ml.1

install_config:
	mkdir -p ~/.config/ml
	sed "s!utils/weights.bin!${CFGPREFIX}/ml.weights!" utils/settings.cfg > ${CFGPREFIX}/ml.cfg

uninstall:
	rm -v $(BINPREFIX)/${BIN}
	rm -v $(MANPREFIX)/man1/ml.1

man: build
	help2man -N ./ml -I doc/man.txt > doc/ml.1

run: build
	@./${BIN} train data/sample_data.json | tee data/train_history.txt
	@./${BIN} predict data/sample_data.json | jq -r '.[] | [values[] as $$val | $$val] | @tsv' > data/net_data.tsv
	@jq -r '.[] | [values[] as $$val | $$val] | @tsv' data/sample_data.json > data/sample_data.tsv
	@gnuplot utils/plot.gpi

debug: build
	gdb -x utils/commands.gdb --tui --args ${BIN} train data/xor.json -e 100
	@#gdb -x utils/commands.gdb --tui --args ${BIN} predict data/sample_data.json

clean:
	@rm $(OBJS) $(OBJDIR) -rv
